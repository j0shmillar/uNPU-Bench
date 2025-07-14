from utils import run_subproc, gen_calibration_table, gen_ds
import numpy as np
import sys
import os

# TODO add batchnorm fusion as an arg
def run_ai8x(model, target_hardware, data_samples, args, ai8x_args):
    for hardware in target_hardware:
        if hardware in ['max78000', 'max78002']:
            if ai8x_args['qat_policy'] is None:
                print("No QAT policy - using PTQ...")
                quantize_cmd = [
                    sys.executable, "ai8x-synthesis/quantize.py",
                    model, model.replace(".pth.tar", "-q8.pth.tar"),
                    "--device", hardware,
                    "--scale", ai8x_args['q_scale'],
                    "-c", ai8x_args['config_file']
                ]
                if ai8x_args['clip_method']:
                    quantize_cmd.extend([
                        "--clip-method", ai8x_args['clip_method']
                    ])


                print("Fusing BatchNorm layers...")
                fuse_cmd = [
                    sys.executable, "ai8x-training/batchnormfuser.py",
                    "-i", model, "-o", model,
                    "-oa", args.model_module_name
                ]
                run_subproc(fuse_cmd, "BatchNorm fusing failed.")
                print(fuse_cmd)
                print("BatchNorm fusion complete.")
                print("Quantizing...")
                run_subproc(quantize_cmd, "Quantization failed.")
                print(quantize_cmd)
                print("Quantization complete.")

            model = model.replace(".pth.tar", "-q8.pth.tar")

            ai8xize_args = [sys.executable, "ai8x-synthesis/ai8xize.py"]
            for arg, value in ai8x_args.items():
                if arg == "q_scale":
                    continue
                if arg == "no-pipeline" or arg == "max-speed":
                    if hardware != "max78002":
                        print(f"Invalid argument {arg} for MAX78000")
                        return None
                if value is True:
                    ai8xize_args.append(f"--{arg.replace('_', '-')}")
                elif value not in [None, False]:
                    ai8xize_args.extend([f"--{arg.replace('_', '-')}", str(value)])

            ai8xize_args.append(f"--checkpoint-file={model}")
            # TODO support using multiple data samples i.e. all given
            np.save("data_sample_int.npy", np.load(data_samples[0])[0].astype(np.int64)) # TODO delete after use (also, should support multiple data samples, data samples of shape (NHWC) (i.e. current) AND (HWC), etc)
            ai8xize_args.append(f"--sample-input=data_sample_int.npy")
            ai8xize_args.append(f"--device={hardware.upper()}")

            print(ai8xize_args)

            print("Generating AI8X model and code...")
            res = run_subproc(ai8xize_args, "AI8X model/code gen failed.")
            # TODO either return 1 / None, or output file path - don't mix and match
            if res is not None:
                return 1
            else:
                return None

# TODO 
# - fix
# - rename 2 TPU-MLIR?
def run_cvi(onnx_path, args):
    """Compile model using CV180X toolchain."""
    model_name = os.path.basename(onnx_path).replace('.onnx', '')
    model_mlir = onnx_path.replace('.onnx', '.mlir')
    output_path = onnx_path.replace('.onnx', '.cvimodel')
    cal_table = args["calibration_table"] or gen_calibration_table(".table")
    data_samples = args["data_samples"].split(',') # TODO avoid repetition, pass formatted list from convert/main

    transform_cmd = [
        "tpu-mlir/model_transform.py",
        "--model_name", model_name,
        "--model_def", onnx_path,
        "--mlir", model_mlir,
        "--input_shape", args["input_shape"],
        "--output_names", args["output_names"],
        "--test_input", data_samples[0] # TODO support using multiple data samples
    ]

    optional_args = {
        "--model_data": args["caffe_model"],
        "--resize_dims": args["resize_dims"],
        "--pixel_format": args["pixel_format"],
        "--test_result": args["val_result_file"],
        "--excepts": args["excepts"],
    }

    for k, v in optional_args.items():
        if v: transform_cmd += [k, v]
    if args.keep_aspect_ratio: transform_cmd.append("--keep_aspect_ratio")
    if args.debug: transform_cmd.append("--debug")

    run_subproc(transform_cmd, "Transform step failed")
    os.chdir(os.path.dirname(onnx_path))

    dataset_path = args.data_samples or gen_ds(args.input_shape[1:])
    cali_cmd = [
        "run_calibration.py",
        model_mlir,
        "--dataset", dataset_path,
        "--input_num", str(args.input_shape[0]),
        "-o", cal_table
    ]
    run_subproc(cali_cmd, "Calibration failed")

    tol1, tol2 = 0.84, 0.45
    if args.tolerance:
        split = args.tolerance.split(',')
        tol1 = float(split[0])
        tol2 = float(split[1]) if len(split) > 1 else tol2

    deploy_cmd = [
        "model_deploy.py",
        "--mlir", model_mlir,
        "--quantize", args["quantize"],
        "--calibration_table", cal_table,
        "--processor", args["target_hardware"],
        "--tolerance", str(tol1),
        "--correctness", str(tol2),
        "--model", output_path
    ]
    if args.dynamic: deploy_cmd.append("--dynamic")
    if args.excepts: deploy_cmd += ["--excepts", args.excepts]
    if args.debug: deploy_cmd.append("--debug")

    run_subproc(deploy_cmd, "Deploy failed")
    os.chdir("..")
    return output_path


def run_vela(out_dir, args):
    """Compile a TFLite model using Vela."""
    tflite_model = os.path.join(out_dir, f"{args.model_name}_full_integer_quant.tflite")
    vela_cmd = [
        "vela",
        "--accelerator-config", args.target_hardware,
        "--recursion-limit", str(args.recursion_limit),
        "--optimise", args.optimise,
        tflite_model,
        "--output-dir", out_dir
    ]

    config_flags = {
        "--config": args.config_vela,
        "--system-config": args.config_vela_system,
        "--memory-mode": args.memory_mode,
        "--tensor-allocator": args.tensor_allocator,
        "--max-block-dependency": args.max_block_dependency,
        "--arena-cache-size": args.arena_cache_size,
        "--cpu-tensor-alignment": args.cpu_tensor_alignment,
        "--hillclimb-max-iterations": args.hillclimb_max_iterations,
    }
    for k, v in config_flags.items():
        if v is not None:
            vela_cmd.extend([k, str(v)])
    if args.force_symmetric_int_weights:
        vela_cmd.append("--force-symmetric-int-weights")

    run_subproc(vela_cmd, "Vela compilation failed")
    return out_dir
    
# TODO need to loop over target hardware and run cmd for each of supported by run_eiq
# TODO rename to 'neutron'?
def run_eiq(tflm_model, target_hardware, eiq_args):
    """Run NXP's Neutron compiler."""
    eiq_cmd = [
        eiq_args['eiq_path'],
        "--input", tflm_model,
        "--output", "out_path", # TODO softcode
        "--custom-options", f"target {target_hardware}"
    ]
    run_subproc(eiq_cmd, "Neutron compile failed")
    return "out_path"
