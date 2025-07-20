from utils import run_subproc, gen_calibration_table, gen_ds
import numpy as np
import sys
import os

def run_ai8x(model, target_hardware, data_sample, ai8x_args, args):
    if target_hardware in ['max78000', 'max78002']:
        print("Fusing BatchNorm layers...")
        fuse_cmd = [
            sys.executable, "ai8x-training/batchnormfuser.py",
            "-i", model, "-o", f"{model}_bn",
            "-oa", args.model_module_name
        ]
        res = run_subproc(fuse_cmd, "BatchNorm fusion failed.")
        if res is None:
            return None
        print("BatchNorm fusion complete.")
        if ai8x_args['qat_policy'] is None and args.bit_width==8:
            print("No QAT policy, bit width set to INT8 - using PTQ.")
            quantize_cmd = [
                sys.executable, "ai8x-synthesis/quantize.py",
                f"{model}_bn", f"{model}_bn",
                "--device", target_hardware,
                "--scale", ai8x_args['q_scale'],
                "-c", ai8x_args['config_file']
            ]
            if ai8x_args['clip_method']:
                quantize_cmd.extend(["--clip-method", ai8x_args['clip_method']])
            print("Quantizing...")
            res = run_subproc(quantize_cmd, "Quantization failed.")
            if res is None:
                return None

        ai8xize_args = [sys.executable, "ai8x-synthesis/ai8xize.py"]
        for arg, value in ai8x_args.items():
            if arg == "q_scale":
                continue
            if arg in ["no-pipeline", "max-speed"] and target_hardware != "max78002":
                print(f"Invalid arg {arg} for MAX78000; supported on MAX78002 only.")
                return None
            if value is True:
                ai8xize_args.append(f"--{arg.replace('_', '-')}")
            elif value not in [None, False]:
                ai8xize_args.extend([f"--{arg.replace('_', '-')}", str(value)])

        ai8xize_args.append(f"--checkpoint-file={model}")

        data = np.load(data_sample)
        if data.ndim == 4:
            data = data[0]
        temp_path = "ds_sample.npy"
        np.save(temp_path, data.astype(np.int64))
        ai8xize_args.append(f"--sample-input={temp_path}")
        ai8xize_args.append(f"--device={target_hardware.upper()}")
        res = run_subproc(ai8xize_args, "ai8x compiler failed")
        if res is None:
            return None
        os.remove(temp_path)
    print(f"ai8x model/code saved to {args.out_dir}")
    return args.out_dir

# TODO fix
def run_cvi(onnx_path, data_sample, args):
    model_name = "/workspace/src/"+os.path.splitext(os.path.basename(onnx_path))[0]
    model_mlir = "/workspace/src/"+os.path.splitext(onnx_path)[0] + ".mlir"
    output_path = "/workspace/src/"+os.path.splitext(onnx_path)[0] + ".cvimodel"
    cal_table = args.get("calibration_table") or gen_calibration_table(".table") # TODO print / warn user no calibrate table provided so generating one

    input_shapes = args["input_shape"]
    shape_str = input_shapes if isinstance(input_shapes, str) else ",".join(map(str, input_shapes))

    transform_cmd = [
        "model_transform.py",
        "--model_name", model_name,
        "--model_def", onnx_path,
        "--mlir", model_mlir,
        "--input_shape", shape_str,
        "--output_names", args["output_names"],
    ]

    data = np.load(data_sample)
    if data.ndim == 4:
        data = data[0]
    os.makedirs("/workspace/src/temp", exist_ok=True)
    temp_path = "/workspace/src/temp/ds_sample.npy" 
    np.save(temp_path, data.astype(np.int64))

    optional_args = {
        "--model_data": args.get("caffe_model"),
        "--resize_dims": args.get("resize_dims"),
        "--pixel_format": args.get("pixel_format"),
        "--excepts": args.get("excepts")}

    for k, v in optional_args.items():
        if v:
            transform_cmd += [k, v]
    if args.get("keep_aspect_ratio"):
        transform_cmd.append("--keep_aspect_ratio")
    if args.get("debug"):
        transform_cmd.append("--debug")

    res = run_subproc(transform_cmd, "cvi transform failed")
    if res is None:
        return None
    os.chdir(os.path.dirname(onnx_path))

    cali_cmd = [
        "run_calibration.py",
        model_mlir,
        "--dataset", temp_path,
        "--input_num", str(args["input_shape"][0]),
        "-o", cal_table
    ]
    res = run_subproc(cali_cmd, "cvi calibration failed")
    if res is None:
        return None

    quant = {8: "INT8", 16: "F16", 32: "F32"}
    # support all supported hardware formats + bit widths?
    deploy_cmd = [
        "model_deploy.py",
        "--mlir", model_mlir,
        "--quantize", quant[args["bit_width"]],
        "--calibration_table", cal_table,
        "--processor", args["target_hardware"],
        "--tolerance", str(args["tolerance"]),
        "--model", output_path
    ]
    if args.get("dynamic"):
        deploy_cmd.append("--dynamic")
    if args.get("excepts"):
        deploy_cmd += ["--excepts", args["excepts"]]
    if args.get("debug"):
        deploy_cmd.append("--debug")

    res = run_subproc(deploy_cmd, "cvi compilation failed")
    if res is None:
        return None
    
    os.chdir("..")
    os.remove(temp_path)

    return output_path

def run_vela(out_name, args):
    quant_suffix = {
        8: "_full_integer_quant.tflite",
        16: "_float16.tflite",
        32: "_float32.tflite"
    }.get(args.get("bit_width", 8), "_full_integer_quant.tflite")

    tflite_model = f"{out_name}{quant_suffix}"
    
    ethos_hardware = ["hxwe2", "ethos-u55-32", "ethos-u55-64", "ethos-u55-128", "ethos-u55-256", "ethos-u65-256", "ethos-u65-512"]

    if args["target_hardware"] == "hxwe2":
        args["target_hardware"] = "ethos-u55-64"
    if args["target_hardware"] in ethos_hardware:
        vela_cmd = [
            "vela",
            "--accelerator-config", args["target_hardware"],
            "--recursion-limit", str(args["recursion_limit"]),
            "--optimise", args["vela_optimise"],
            tflite_model,
            "--output-dir", os.path.dirname(out_name)
        ]

        config_flags = {
            "--config": args.get("config_vela"),
            "--system-config": args.get("config_vela_system"),
            "--memory-mode": args.get("memory_mode"),
            "--tensor-allocator": args.get("tensor_allocator"),
            "--max-block-dependency": args.get("max_block_dependency"),
            "--arena-cache-size": args.get("arena_cache_size"),
            "--cpu-tensor-alignment": args.get("cpu_tensor_alignment"),
            "--hillclimb-max-iterations": args.get("hillclimb_max_iterations"),
        }

        for k, v in config_flags.items():
            if v is not None:
                vela_cmd.extend([k, str(v)])
        if args.get("force_symmetric_int_weights"):
            vela_cmd.append("--force-symmetric-int-weights")

        res = run_subproc(vela_cmd, "ARM Vela compiler failed.")
        if res is None:
            return None
    else:
        print(f"Hardware platform {args['target_hardware']} not supported by ARM Vela.")
        return None
    return tflite_model


def run_eiq(base_name, target_hardware, model_name, eiq_args):
    tflm_model = f"{base_name}/{model_name}_full_integer_quant_tflm.tflite"
    eiq_model = f"{base_name}/{model_name}_full_integer_quant_eiq.tflite"

    eiq_cmd = [
        eiq_args['eiq_path'],
        "--input", tflm_model,
        "--output", eiq_model,
        "--custom-options", f"target {target_hardware}"]
    res = run_subproc(eiq_cmd, "eIQ compiler failed.")
    if res is None:
        return None
    return eiq_model
