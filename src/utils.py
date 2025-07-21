import os
import sys
import torch
import onnx2tf
import tempfile
import importlib
import subprocess
import numpy as np

# reg custom op for ONNX export
def exp2_symbolic(g, input):
    log2 = g.op("Constant", value_t=torch.tensor(2.0).log())
    return g.op("Exp", g.op("Mul", input, log2))

torch.onnx.register_custom_op_symbolic("aten::exp2", exp2_symbolic, opset_version=12)

def torch2onnx(model, pth_path, args, out_dir):
    checkpoint = torch.load(pth_path, map_location='cpu')
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if not state_dict:
        print("No state_dict found.")
        return None

    model.eval()

    base_name = args["model_name"] + ".onnx"
    onnx_path = os.path.join(out_dir, base_name)

    dummy_input = torch.randn(*args["input_shape"])

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=args.get("export_params", True),
            do_constant_folding=args.get("do_constant_folding", True),
            input_names=args["input_names"],
            output_names=args["output_names"],
            opset_version=args.get("opset", 12),
            dynamic_axes=args.get("dynamic_axes", None),
            keep_initializers_as_inputs=args.get("keep_initializers_as_inputs", False),
            custom_opsets=args.get("custom_opsets", None))
        print(f"✅ Saved ONNX model to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

def onnx2tflm(onnx_path, args):
    sample = np.load(args["data_sample"])
    print(f"Loaded sample shape: {sample.shape}")

    if sample.ndim == 4:
        sample = sample[0] 
        print(f"Removed batch dim, new shape: {sample.shape}")
    elif sample.ndim == 3:
        print(f"Sample already 3D: {sample.shape}")
    else:
        raise ValueError(f"Expected 3D or 4D sample, got shape {sample.shape}")

    layout = args["input_layout"]

    if layout == "NCHW":
        if sample.shape[0] > 10: 
            print("Warning: sample might already be in HWC format?")
        if sample.shape != (3, 32, 32):
            print(f"Unexpected shape for NCHW input: {sample.shape}")
        if sample.ndim == 3:
            sample = sample.transpose(1, 2, 0)  # CHW -> HWC
            print(f"Transposed to HWC: {sample.shape}")
        else:
            raise ValueError(f"Expected 3D sample for NCHW, got shape {sample.shape}")

    elif layout == "NCW":
        if sample.ndim == 3:
            sample = sample.transpose(1, 2, 0)  # e.g., NCW to CWN
            print(f"Transposed to match NCW layout: {sample.shape}")
        elif sample.ndim == 2:
            sample = sample.transpose(1, 0)
            print(f"Transposed 2D: {sample.shape}")
        else:
            raise ValueError(f"Unsupported shape for NCW input layout: {sample.shape}")

    if sample.ndim != 3:
        raise ValueError(f"Expected 3D sample before reshaping, got {sample.shape}")
    
    N, W, C = sample.shape
    sample = sample.reshape((1, 1, N, W, C))
    
    temp_sample_path = "sample_rs.npy"
    np.save(temp_sample_path, sample.astype(float))

    mean, std = sample.mean(), sample.std()

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=os.path.dirname(onnx_path),
        output_integer_quantized_tflite=args["output_integer_quantized_tflite"],
        not_use_onnxsim=not args["use_onnxsim"],
        verbosity="debug", 
        quant_type=args["tflm_quant_type"],
        disable_group_convolution=args["disable_group_convolution"],
        enable_batchmatmul_unfold=args["enable_batchmatmul_unfold"],
        custom_input_op_name_np_data_path=[[args["input_names"][0], temp_sample_path, mean, std]])

    os.remove(temp_sample_path)

    print(f"✅ Saved TFLM model to {os.path.dirname(onnx_path)}")
    return os.path.dirname(onnx_path)

def setup_ai8x(device_id=85, use_8bit=True):
    try:
        ai8x_root = os.environ.get("AI8X_TRAIN_PATH")
        if ai8x_root not in sys.path:
            sys.path.insert(0, ai8x_root)
        import ai8x
        ai8x.set_device(device_id, 0, use_8bit)
        return ai8x
    except ImportError as e:
        sys.exit("\033[91mai8x module not found. Make sure AI8X_TRAIN_PATH is set correctly.\033[0m")

def get_model_from_name(path, class_name, args=None):
    setup_ai8x()
    module_path = path.replace('.py', '').replace('/', '.')
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        sys.exit(f"\033[91mModule not found: {module_path}\033[0m")
    model_class = getattr(module, class_name)
    return model_class(*(args or []))

def gen_ds(shape, count=10):
    out_dir = tempfile.mkdtemp(prefix="random_dataset_")
    for i in range(count):
        sample = np.random.rand(*shape).astype(np.float32)
        np.save(os.path.join(out_dir, f"sample_{i}.npy"), sample)
    print(f"Dataset saved to {out_dir}")
    return out_dir

def gen_calibration_table(out_path):
    with open(out_path, "w") as f:
        f.write("# Calibration Table\ninput_scale: 1.0\ninput_zero_point: 0\n")
    print(f"Calibration table saved to {out_path}")
    return out_path

def make_out_dir(args):
    out_dir = args.out_dir

    def is_empty(path):
        return not os.path.exists(path) or not os.listdir(path)

    if args.overwrite:
        os.makedirs(out_dir, exist_ok=True)
        print(f"📝 Using output directory (overwrite enabled): {out_dir}")
    else:
        if is_empty(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            print(f"📁 Using output directory: {out_dir}")
        else:
            base_backup = out_dir + "_backup"
            backup_dir = base_backup
            i = 1
            while os.path.exists(backup_dir):
                backup_dir = f"{base_backup}_{i}"
                i += 1

            # shutil.move(out_dir, backup_dir)
            os.makedirs(backup_dir)
            print(f"⚠️  Output directory was not empty. Created: {backup_dir}")
            out_dir = backup_dir

    args.out_dir = out_dir
    return args

def run_subproc(command, debug, error_msg):
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if debug:
            for line in process.stdout:
                print(line, end="")  
            process.wait()
        if process.returncode != 0:
            print(f"{error_msg} (exit code {process.returncode})")
            return None
        return True
    except Exception as e:
        print(f"{error_msg}: {e}")
        return None
