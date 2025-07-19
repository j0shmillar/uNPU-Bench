import sys
import argparse
from utils import get_model_from_name
from parse import compile
from flags import SUPPORTED_BIT_WIDTHS, SUPPORTED_HARDWARE, GLOBAL_FLAGS, PLATFORM_FLAGS

#TODO
# check eiq & cvi
# add proper input validation with YAML (see chatgpt)
# fix all_scripts (+ rename)
# reformat files with GPT (make it easy to add new chips, etc)
# & Add README

# TODO for later
# test generated code runs on device + tidy device templates
# maybe fix WARNING: axes don't match array

def val_bitwidth(target_formats, target_hardware, bit_width):
    for fmt in target_formats:
        if fmt in SUPPORTED_BIT_WIDTHS:
            if bit_width not in SUPPORTED_BIT_WIDTHS[fmt]:
                raise ValueError(f"Bit-width {bit_width} not supported for {fmt} (supported: {SUPPORTED_BIT_WIDTHS[fmt]})")
    for hw in target_hardware:
        if hw not in SUPPORTED_HARDWARE:
            raise ValueError(f"'{hw}' is not supported (supported: {SUPPORTED_HARDWARE})")

def val_flags(target_hardware, target_formats):
    specified_flags = {
        arg.lstrip('-').replace('-', '_')
        for arg in sys.argv[1:]
        if arg.startswith('--')}

    platform_to_flags = {}
    for fmt in target_formats:
        for hw in target_hardware:
            platform_key = hw.lower() if hw.lower() in PLATFORM_FLAGS else fmt.lower()
            if platform_key in PLATFORM_FLAGS:
                flags = PLATFORM_FLAGS[platform_key]["flags"]
                platform_to_flags.setdefault(platform_key, set()).update(flags)

    flag_to_platforms = {}
    for platform, flags in platform_to_flags.items():
        for flag in flags:
            flag_to_platforms.setdefault(flag, set()).add(platform)

    for flag in specified_flags:
        if flag in GLOBAL_FLAGS:
            continue

        supported_platforms = flag_to_platforms.get(flag, set())
        target_platforms = {
            hw.lower() if hw.lower() in PLATFORM_FLAGS else fmt.lower()
            for fmt in target_formats
            for hw in target_hardware
        }

        unsupported = target_platforms - supported_platforms

        if not supported_platforms:
            print(f"Note: '{flag}' isn't supported on the chosen platforms ({', '.join(target_platforms)}) - it will be ignored.")
        elif unsupported:
            print(f"Note: '{flag}' is only supported on {sorted(supported_platforms)}, and will be ignored for: {sorted(unsupported)}")

def parse():
    parser = argparse.ArgumentParser(description="µNPU Universal Compiler Wrapper", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--model_ckpt', required=True, help='Path to checkpoint file')
    parser.add_argument('--model_name', required=True, help='Model name')
    parser.add_argument('--model_module_name', required=True, help='Model module name')
    parser.add_argument("--model_module_args", type=str, help="Any required model-specific arguments")

    parser.add_argument('--target_format', required=True, help='Comma-separated target formats: ai8x,tflm,vela,onnx')
    parser.add_argument('--target_hardware', required=False, default='max78000', help='Comma-separated hardware: max78000,ethos-u55-128,...')

    parser.add_argument('--data_sample', required=True, help='.npy dataset for quantization')
    parser.add_argument('--input_names', required=True, help='Comma-separated model input names')
    parser.add_argument('--input_shape', type=int, nargs='+', required=True, help='Input shape, e.g. 1 3 224 224')
    parser.add_argument('--input_layout', choices=['NCHW', 'NHWC', 'NCW', 'NWC'], default='NCHW', help='Input layout')
    parser.add_argument('--output_names', required=True, help='Comma-separated model output names')
    parser.add_argument('--output_shape', type=int, nargs='+', required=True, help='Output shape, e.g. 1 3 224 224')
    parser.add_argument('--bit_width', type=int, default=8, help='Quantization bit-width')

    parser.add_argument('--out_dir', required=True, type=str, help='Output directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output if out_dir exists')

    # ONNX
    parser.add_argument('--opset', required=False, default=19, help='ONNX opset')
    parser.add_argument('--custom_opsets', required=False, default=None, help='Custom ONNX opset/ops (format: KEY (str): opset domain name VALUE (int): opset version)')
    parser.add_argument('--export_params', required=False, default=True, help='ONNX export params')
    parser.add_argument('--do_constant_folding', required=False, default=True, help='Do constant folding')
    parser.add_argument('--keep_initializers_as_inputs', required=False, default=False, help='Add initializer to exported weights')
    parser.add_argument('--dynamic_axes', required=False, default=None, help='Schema for dynamic input/output shape')
    
    # TFLM
    parser.add_argument('--use_onnxsim', default=False, action='store_true', help='Use onnxsim for TFLM')
    parser.add_argument('--output_integer_quantized_tflite', default=True, action='store_true', help='INT8 quantized TFLM')
    parser.add_argument('--tflm_quant_type', type=str, choices=["per_tensor", "per_channel"], default='per_channel', help='TFLM quantization method')
    parser.add_argument('--disable_group_convolution', default=False, action='store_true', help='Disable group convolution')
    parser.add_argument('--enable_batchmatmul_unfold', default=False, action='store_true', help='Enable batchmatmul unfold')

    # ai8x
    parser.add_argument('--avg_pool_rounding', action='store_true', help='Round average pooling results')
    parser.add_argument('--simple1b', action='store_true', help='Use simple XOR instead of 1-bit multiplication')
    parser.add_argument('--config_file', type=str, help='YAML model config file path')
    parser.add_argument('--board_name', type=str, default='EvKit_V1', help='Target board')
    parser.add_argument('--compact_weights', action='store_true', help='Use memcpy for weights')
    parser.add_argument('--mlator', action='store_true', help='Use hardware byte swap')
    parser.add_argument('--softmax', action='store_true', help='Add software softmax')
    parser.add_argument('--boost', type=float, help='Boost CNN supply voltage')
    parser.add_argument('--no_wfi', action='store_true', help='Disable WFI instructions')
    parser.add_argument('--max_speed', action='store_true', help='Prioritize speed')
    parser.add_argument('--fifo', action='store_true', help='Use FIFO for streaming')
    parser.add_argument('--fast_fifo', action='store_true', help='Use fast FIFO')
    parser.add_argument('--fast_fifo_quad', action='store_true', help='Use fast FIFO in quad mode')
    parser.add_argument('--riscv', action='store_true', help='Use RISC-V')
    parser.add_argument('--riscv_exclusive', action='store_true', help='Exclusive SRAM for RISC-V')
    parser.add_argument('--input_offset', type=str, help='First layer input offset (hex)')
    parser.add_argument('--write_zero_registers', action='store_true', help='Write zero registers')
    parser.add_argument('--init_tram', action='store_true', help='Init TRAM to 0')
    parser.add_argument('--zero_sram', action='store_true', help='Zero SRAM')
    parser.add_argument('--zero_unused', action='store_true', help='Zero unused registers')
    parser.add_argument('--max_verify_length', type=int, help='Max output verify length')
    parser.add_argument('--no_unload', action='store_true', help='No cnn_unload()')
    parser.add_argument('--no_deduplicate_weights', action='store_true', help='No weight deduplication')
    parser.add_argument('--no_scale_output', action='store_true', help='Do not scale output')
    parser.add_argument('--qat_policy', type=str, default=None, help='QAT policy')
    parser.add_argument('--clip_method', type=str, choices=['AVGMAX', 'MAX', 'STDDEV', 'SCALE'], default=None, help='Clip method for quantization')
    parser.add_argument('--q_scale', type=str, default="0.85", help='Quantization scale')

    # Vela
    parser.add_argument('--config_vela', type=str, help='Vela config file')
    parser.add_argument('--config_vela_system', type=str, default="internal-default", help='Vela system config')
    parser.add_argument('--force_symmetric_int_weights', action='store_true', help='Force symmetric int weights')
    parser.add_argument('--memory_mode', type=str, default="internal-default", help='Memory mode')
    parser.add_argument('--tensor_allocator', choices=['LinearAlloc', 'Greedy', 'HillClimb'], default='HillClimb', help='Tensor allocator')
    parser.add_argument('--max_block_dependency', type=int, choices=[0, 1, 2, 3], default=3, help='Max block dependency')
    parser.add_argument('--arena_cache_size', type=int, help='Arena cache size')
    parser.add_argument('--cpu_tensor_alignment', type=int, default=16, help='CPU tensor alignment')
    parser.add_argument('--recursion_limit', type=int, default=1000, help='Recursion limit')
    parser.add_argument('--hillclimb_max_iterations', type=int, default=99999, help='HillClimb iterations')
    parser.add_argument('--vela_optimise', choices=['Size', 'Performance'], default='Performance', help='Optimisation strategy')

    # eIQ
    parser.add_argument("--eiq_path", type=str, default="/opt/nxp/eIQ_Toolkit_v1.13.1/bin/neutron-converter/MCU_SDK_2.16.000/neutron-converter", help="eIQ Neutron SDK path")

    # CVI
    parser.add_argument("--calibration_table", type=str, help="Quantization table path. Required when using INT8 quantization for CVI")
    parser.add_argument("--tolerance", type=float, help="Tolerance for minimum similarity between MLIR quantized and MLIR FP32 inference results")
    parser.add_argument("--correctness", type=str, default="0.99,0.90", help="Tolerance for minimum similarity between simulator and MLIR quantized inference results. Default: '0.99,0.90'")
    parser.add_argument("--dynamic", action="store_true", help="Support dynamic input shapes")

    return parser

def main():
    parser = parse()
    args = parser.parse_args()

    target_formats = [fmt.strip() for fmt in args.target_format.split(',')]
    target_hardware = [hw.strip() for hw in args.target_hardware.split(',')]
    input_names = [nm.strip() for nm in args.input_names.split(',')]
    output_names = [nm.strip() for nm in args.output_names.split(',')]

    try:
        val_bitwidth(target_formats, target_hardware, args.bit_width)
        val_flags(target_hardware, target_formats)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    model = get_model_from_name(args.model, args.model_module_name, args.model_module_args)

    compile(
        model=model,
        model_name=args.model_name,
        model_ckpt=args.model_ckpt,
        target_formats=target_formats,
        target_hardware=target_hardware,
        data_sample=args.data_sample,
        input_names=input_names,
        output_names=output_names,
        args=args)

if __name__ == "__main__":
    main()
