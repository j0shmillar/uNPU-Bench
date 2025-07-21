# TODO for later
# test generated code runs on device + tidy device templates
# add code compilation instructions to README
# update .gitignore and .dockerignore
# rename to frdmmcxn947
# TIDY CODE UNDER NPU
# ADD OPS GEN FOR NXPMCXN947

# TODO
# add more quant ops for TFLM, etc

import sys
import warnings
import torch.nn as nn

from parse import compile
from utils import get_model_from_name
from val import load_platform_spec, argval

TYPE_MAP = {"int": int, "float": float, "str": str, "bool": bool}

def resolve_format_dependencies(target_formats, spec):
    resolved = set()

    def dfs(fmt):
        if fmt in resolved:
            return
        resolved.add(fmt)
        deps = spec["formats"].get(fmt, {}).get("depends_on", [])
        for dep in deps:
            dfs(dep)

    for fmt in target_formats:
        dfs(fmt)

    return list(resolved)

def add_dynamic_flags_from_yaml(parser, target_formats, spec):
    for fmt in target_formats:
        fmt_spec = spec["formats"].get(fmt, {})
        flags = fmt_spec.get("flags", {})
        for flag_name, flag_props in flags.items():
            argname = f"--{flag_name}"
            kwargs = {}
            for key, val in flag_props.items():
                if key == "type":
                    if isinstance(val, str) and val in TYPE_MAP:
                        kwargs["type"] = TYPE_MAP[val]
                    else:
                        raise ValueError(f"Unknown or unsafe type '{val}' for flag '{flag_name}'")
                elif key == "action":
                    kwargs["action"] = val
                elif key == "default":
                    kwargs["default"] = val
                elif key == "choices":
                    kwargs["choices"] = val
                elif key == "required":
                    kwargs["required"] = val
                elif key == "help":
                    kwargs["help"] = val

            parser.add_argument(argname, **kwargs)

def parse():
    import argparse
    parser = argparse.ArgumentParser(description="µNPU Universal Compiler Wrapper", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--target_format', required=True)
    parser.add_argument('--target_hardware', required=False, default='max78000')

    args, _ = parser.parse_known_args()
    user_formats = [fmt.strip() for fmt in args.target_format.split(',')]

    spec = load_platform_spec()

    all_formats = resolve_format_dependencies(user_formats, spec)
    add_dynamic_flags_from_yaml(parser, all_formats, spec)

    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--model_module_name', type=str, required=True, help='Model module name')
    parser.add_argument('--model_module_args', type=str, help='Model-specific args')
    parser.add_argument('--data_sample', type=str, required=True, help='.npy dataset for quantization')
    parser.add_argument('--input_names', type=str, required=True, help='Comma-separated input names')
    parser.add_argument('--input_shape', type=int, nargs='+', required=True, help='Input shape')
    parser.add_argument('--input_layout', choices=['NCHW', 'NHWC', 'NCW', 'NWC'], default='NCHW')
    parser.add_argument('--output_names', type=str, required=True, help='Comma-separated output names')
    parser.add_argument('--output_shape', type=int, nargs='+', required=True, help='Output shape')
    parser.add_argument('--bit_width', type=int, default=8, help='Quantization bit-width')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--debug', type=bool, default=False, help='Debug subprocess outputs')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite outputs')

    return parser


def main():
    parser = parse()
    args = parser.parse_args()

    target_formats = [fmt.strip() for fmt in args.target_format.split(',')]
    target_hardware = [hw.strip() for hw in args.target_hardware.split(',')]
    args.input_names = [nm.strip() for nm in args.input_names.split(',')]
    args.output_names = [nm.strip() for nm in args.output_names.split(',')]

    spec = load_platform_spec("platforms.yaml")
    errors = argval(args, target_formats, target_hardware, spec)
    if errors:
        print("\n❌ Configuration error:")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    model = get_model_from_name(args.model, args.model_module_name, args.model_module_args)

    def has_batchnorm(m):
        for module in m.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return True
        return False

    has_bn = has_batchnorm(model)
    hw_set = set(hw.lower() for hw in target_hardware)
    is_ai8x = any(hw in {"max78000", "max78002"} for hw in hw_set)

    if has_bn and is_ai8x and len(hw_set) > 1:
        other_targets = hw_set - {"ai8x", "max78000"}
        warnings.warn(
            f"\033[93mBatchNorm layers detected. These will be removed for ai8x/MAX78000/2 as not supported, "
            f"but retained for: {', '.join(other_targets)}\033[0m"
        )

    result = compile(
        model=model,
        model_name=args.model_name,
        model_ckpt=args.model_ckpt,
        target_formats=target_formats,
        target_hardware=target_hardware,
        data_sample=args.data_sample,
        input_names=args.input_names,
        output_names=args.output_names,
        args=args)

    if not result:
        sys.exit(1)

if __name__ == "__main__":
    main()

