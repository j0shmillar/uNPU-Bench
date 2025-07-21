from utils import run_subproc

import os

def export(out_name, args):
    quant_suffix = {
        8: "_full_integer_quant.tflite",
        16: "_float16.tflite",
        32: "_float32.tflite"
    }.get(args.get("bit_width", 8), "_full_integer_quant.tflite")

    tflite_model = f"{out_name}{quant_suffix}"

    if args["target_hardware"] == "hxwe2":
        args["target_hardware"] = "ethos-u55-64"

    if args["target_hardware"] not in {
        "ethos-u55-32", "ethos-u55-64", "ethos-u55-128",
        "ethos-u55-256", "ethos-u65-256", "ethos-u65-512"
    }:
        print(f"Hardware platform {args['target_hardware']} not supported by ARM Vela.")
        return None

    vela_cmd = [
        "vela",
        "--accelerator-config", args["target_hardware"],
        "--recursion-limit", str(args["recursion_limit"]),
        "--optimise", args["vela_optimise"],
        tflite_model,
        "--output-dir", os.path.dirname(out_name)]

    config_flags = {
        "--config": args.get("config_vela"),
        "--system-config": args.get("config_vela_system"),
        "--memory-mode": args.get("memory_mode"),
        "--tensor-allocator": args.get("tensor_allocator"),
        "--max-block-dependency": args.get("max_block_dependency"),
        "--arena-cache-size": args.get("arena_cache_size"),
        "--cpu-tensor-alignment": args.get("cpu_tensor_alignment"),
        "--hillclimb-max-iterations": args.get("hillclimb_max_iterations")}

    for k, v in config_flags.items():
        if v is not None:
            vela_cmd.extend([k, str(v)])
    if args.get("force_symmetric_int_weights"):
        vela_cmd.append("--force-symmetric-int-weights")

    if run_subproc(vela_cmd, args['debug'], "ARM Vela compiler failed.") is None:
        return None

    return tflite_model