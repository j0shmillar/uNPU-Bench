from utils import run_subproc

import os
import sys
import numpy as np

def export(model, target_hardware, data_sample, ai8x_args, args):
    if target_hardware in ['max78000', 'max78002']:
        print("Fusing BatchNorm layers...")
        fuse_cmd = [
            sys.executable, "ai8x-training/batchnormfuser.py",
            "-i", model, "-o", f"{model}_bn",
            "-oa", args.model_module_name]
        
        if run_subproc(fuse_cmd, args.debug, "BatchNorm fusion failed.") is None:
            return None

        if ai8x_args['qat_policy'] is None and args.bit_width == 8:
            print("No QAT policy, bit width set to INT8 - using PTQ.")
            quantize_cmd = [
                sys.executable, "ai8x-synthesis/quantize.py",
                f"{model}_bn", f"{model}_bn",
                "--device", target_hardware,
                "--scale", ai8x_args['q_scale'],
                "-c", ai8x_args['config_file']]
            
            if ai8x_args['clip_method']:
                quantize_cmd.extend(["--clip-method", ai8x_args['clip_method']])
            print("Quantizing...")
            if run_subproc(quantize_cmd, args.debug, "Quantization failed") is None:
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

        model_bn = f"{model}_bn"
        ai8xize_args.append(f"--checkpoint-file={model_bn}")

        data = np.load(data_sample)
        if data.ndim == 4:
            data = data[0]
        temp_path = "ds_sample.npy"
        np.save(temp_path, data.astype(np.int64))

        ai8xize_args.extend([
            f"--sample-input={temp_path}",
            f"--device={target_hardware.upper()}"])
        
        if run_subproc(ai8xize_args, args.debug, "ai8x compiler failed") is None:
            return None
        os.remove(temp_path)

    print(f"ai8x model/code saved to {args.out_dir}")
    return args.out_dir