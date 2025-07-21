
import os
import yaml

def load_platform_spec(path="platforms.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def argval(args, target_formats, target_hardware, spec):
    errors = []

    supported_flags = {}
    valid_bit_widths = set()
    format_to_hw = {}

    for fmt in target_formats:
        fmt_spec = spec["formats"].get(fmt)
        if not fmt_spec:
            errors.append(f"Unknown format: {fmt}")
            continue

        valid_bit_widths.update(fmt_spec.get("bit_widths", []))

        allowed_hw = fmt_spec.get("compatible_hardware", [])
        allowed_prefix = fmt_spec.get("requires", {}).get("compatible_hardware_prefix", None)

        compatible = False
        for hw in target_hardware:
            if hw in allowed_hw or (allowed_prefix and hw.startswith(allowed_prefix)):
                compatible = True
                format_to_hw.setdefault(fmt, []).append(hw)

        if not compatible:
            errors.append(f"No compatible hardware found for format '{fmt}' with selected hardware {target_hardware}")

        if "requires" in fmt_spec:
            for field, required in fmt_spec["requires"].items():
                if required and not getattr(args, field, None):
                    errors.append(f"{fmt} requires --{field}")

        for flag, flag_def in fmt_spec.get("flags", {}).items():
            if flag in supported_flags:
                prev_def = supported_flags[flag]
                for key in ['type', 'choices', 'action', 'default']:
                    if key in prev_def or key in flag_def:
                        if prev_def.get(key) != flag_def.get(key):
                            errors.append(f"Flag conflict for --{flag}: {fmt} defines {key}={flag_def.get(key)}, but another format defines {key}={prev_def.get(key)}")
            else:
                supported_flags[flag] = flag_def

    for hw in target_hardware:
        if hw not in spec["hardware"]:
            errors.append(f"Unsupported hardware: {hw}")
            continue

        hw_formats = spec["hardware"][hw].get("formats", [])
        for fmt in target_formats:
            if fmt not in hw_formats:
                errors.append(f"{hw} does not support format: {fmt}")

    if args.bit_width not in valid_bit_widths:
        errors.append(f"bit_width {args.bit_width} not supported across selected formats: {target_formats}")

    if not args.model or not os.path.isfile(args.model):
        errors.append("Invalid or missing model file (--model)")

    if not args.model_ckpt or not os.path.isfile(args.model_ckpt):
        errors.append("Invalid or missing checkpoint (--model_ckpt)")

    if not args.input_names or not args.output_names:
        errors.append("Missing input or output names (--input_names, --output_names)")

    if len(args.input_shape) < 2 or len(args.output_shape) < 1:
        errors.append("Invalid input/output shapes")

    if not os.path.exists(args.data_sample):
        errors.append(f"Data sample not found: {args.data_sample}")

    return errors
