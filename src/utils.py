import os
import sys
import tempfile
import subprocess
import importlib
import numpy as np
import torch
import onnx2tf

# reg custom op for ONNX export
def exp2_symbolic(g, input):
    log2 = g.op("Constant", value_t=torch.tensor(2.0).log())
    return g.op("Exp", g.op("Mul", input, log2))

torch.onnx.register_custom_op_symbolic("aten::exp2", exp2_symbolic, opset_version=12)

def torch2onnx(model, pth_path, args):
    checkpoint = torch.load(pth_path, map_location='cpu')
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if not state_dict:
        print("No state_dict found.")
        return None
    
    model.eval()

    onnx_path = pth_path.replace(".pth.tar", ".onnx")
    dummy_input = torch.randn(*args["input_shape"])

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            do_constant_folding=True,
            input_names=[args["input_names"]],
            output_names=[args["output_names"]]
        )
        print(f"✅ Saved ONNX model to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None
    
def onnx2tflm(onnx_path, args):
    sample = np.load(args["data_samples"][0])
    layout = args["input_layout"]
    if layout == "NCHW":
        sample = sample.transpose(0, 2, 3, 1)
    elif layout == "NCW":
        sample = sample.transpose(0, 2, 1)

    np.save("sample_rs.npy", sample)
    mean, std = sample.mean(), sample.std()

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=os.path.dirname(onnx_path),
        output_integer_quantized_tflite=args["output_integer_quantized_tflite"],
        not_use_onnxsim=not args["use_onnxsim"],
        quant_type=args["tflm_quant_type"],
        disable_group_convolution=args["disable_group_convolution"],
        enable_batchmatmul_unfold=args["enable_batchmatmul_unfold"],
        custom_input_op_name_np_data_path=[[args["input_names"][0], "sample_rs.npy", mean, std]]
    )
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
        print("ai8x module not found. Make sure AI8X_TRAIN_PATH is set correctly.") # TODO make red
        raise e

def get_model_from_name(path, class_name, args=None):
    setup_ai8x() # TODO move
    module_path = path.replace('.py', '').replace('/', '.')
    module = importlib.import_module(module_path)
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

def run_subproc(command, error_msg):
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"{error_msg}: {e.stderr}")
        return None
