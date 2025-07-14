from utils import run_subproc, gen_calibration_table, gen_ds
import numpy as np
import sys
import os

def run_ai8x(model, target_hardware, data_sample, args, ai8x_args):
    for hardware in target_hardware:
        if hardware in ['max78000', 'max78002']:
            if ai8x_args['qat_policy'] is None:
                print("No QAT policy - using PTQ.")
                quantize_cmd = [
                    sys.executable, "ai8x-synthesis/quantize.py",
                    model, model.replace(".pth.tar", "-q8.pth.tar"),
                    "--device", hardware,
                    "--scale", ai8x_args['q_scale'],
                    "-c", ai8x_args['config_file']
                ]
                if ai8x_args['clip_method']:
                    quantize_cmd.extend(["--clip-method", ai8x_args['clip_method']])
                print("Fusing BatchNorm layers...")
                fuse_cmd = [
                    sys.executable, "ai8x-training/batchnormfuser.py",
                    "-i", model, "-o", model,
                    "-oa", args.model_module_name
                ]
                run_subproc(fuse_cmd, "BatchNorm fusion failed.")
                print("BatchNorm fusion complete.")
                print("Quantizing...")
                run_subproc(quantize_cmd, "Quantization failed.")

            model = model.replace(".pth.tar", "-q8.pth.tar")

            ai8xize_args = [sys.executable, "ai8x-synthesis/ai8xize.py"]
            for arg, value in ai8x_args.items():
                if arg == "q_scale":
                    continue
                if arg in ["no-pipeline", "max-speed"] and hardware != "max78002":
                    print(f"Invalid arg {arg} for MAX78000; supported on MAX78002 only.")
                    return None
                if value is True:
                    ai8xize_args.append(f"--{arg.replace('_', '-')}")
                elif value not in [None, False]:
                    ai8xize_args.extend([f"--{arg.replace('_', '-')}", str(value)])

            ai8xize_args.append(f"--checkpoint-file={model}")

            data = np.load(data_sample)
            if data.ndim ==4:  # NHWC → 1xHWC
                data = data[0]
            temp_path = f"ds_sample.npy"
            np.save(temp_path, data.astype(np.int64))
            ai8xize_args.append(f"--sample-input={temp_path}")

            ai8xize_args.append(f"--device={hardware.upper()}")
            run_subproc(ai8xize_args, f"ai8x compiler failed {hardware}.")
            
            os.remove(temp_path)

def run_cvi(onnx_path, data_sample, args):
    model_name = os.path.basename(onnx_path).replace('.onnx', '')
    model_mlir = onnx_path.replace('.onnx', '.mlir')
    output_path = onnx_path.replace('.onnx', '.cvimodel')
    cal_table = args["calibration_table"] or gen_calibration_table(".table")

    input_shapes = args["input_shape"]
    shape_str = input_shapes if isinstance(input_shapes, str) else ",".join(map(str, input_shapes))

    transform_cmd = [
        "tpu-mlir/model_transform.py",
        "--model_name", model_name,
        "--model_def", onnx_path,
        "--mlir", model_mlir,
        "--input_shape", shape_str,
        "--output_names", args["output_names"],
        "--test_input"
    ]

    data = np.load(data_sample)
    if data.ndim ==4:  # NHWC → 1xHWC
        data = data[0]
    temp_path = f"ds_sample.npy"
    np.save(temp_path, data.astype(np.int64))
    transform_cmd.append(temp_path)

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

    run_subproc(transform_cmd, "cvi transform failed.")
    os.chdir(os.path.dirname(onnx_path))

    dataset_path = args.data_sample or gen_ds(args.input_shape[1:])
    cali_cmd = [
        "run_calibration.py",
        model_mlir,
        "--dataset", dataset_path,
        "--input_num", str(args.input_shape[0]),
        "-o", cal_table
    ]
    run_subproc(cali_cmd, "cvi calibration failed.")

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

    run_subproc(deploy_cmd, "cvi compiler failed.")
    os.chdir("..")

    os.remove(temp_path)

    return output_path

def run_vela(out_dir, args):
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

    run_subproc(vela_cmd, "ARM Vela compiler failed.")
    return out_dir

def run_eiq(tflm_model, target_hardware, eiq_args):
    eiq_cmd = [
        eiq_args['eiq_path'],
        "--input", tflm_model,
        "--output", eiq_args["out_dir"],
        "--custom-options", f"target {target_hardware}"
    ]
    run_subproc(eiq_cmd, "eIQ compiler failed.")
    return eiq_args["out_dir"]
