from model_gen import ai8x, vela, eiq, cvi
from code_gen import hxwe2_code_gen, mcxn947_code_gen # TODO update

from pth_to_ckpt import pth_to_pth_tar
from utils import setup_ai8x, torch2onnx, onnx2tflm, make_out_dir

from itertools import product


def get_args(args, keys_with_defaults):
    return {key: getattr(args, key, default) for key, default in keys_with_defaults.items()}

def compile(model, model_name, model_ckpt, target_formats, target_hardware, data_sample, input_names, output_names, args):
    if model_ckpt.endswith(".pth"):
        model_ckpt_tar = model_ckpt + ".tar"
        pth_to_pth_tar(model_ckpt, model_ckpt_tar)
        model_ckpt = model_ckpt_tar

    args = make_out_dir(args)
    onnx_cache = {}

    for fmt, hw in product(target_formats, target_hardware):
        print(f"\n🚀 Compiling for format: {fmt}, hardware: {hw}")

        if fmt in {"onnx", "tflm", "vela", "eiq", "cvi"}:
            if args.bit_width not in onnx_cache:
                onnx_args = get_args(args, {
                    "opset": 19,
                    "opset_version": 19,
                    "custom_opsets": None,
                    "export_params": True,
                    "do_constant_folding": True,
                    "input_names": input_names,
                    "input_shape": args.input_shape,
                    "output_names": output_names,
                    "keep_initializers_as_inputs": False,
                    "dynamic_axes": None,
                    "model_name": args.model_name,
                    "debug": args.debug})
                model_onnx = torch2onnx(model, model_ckpt, onnx_args, args.out_dir)
                if not model_onnx:
                    print(f"❌ ONNX export failed")
                    return None
                onnx_cache[args.bit_width] = model_onnx
            else:
                model_onnx = onnx_cache[args.bit_width]

        if fmt == "onnx":
            continue

        elif fmt == "tflm":
            tflm_args = get_args(args, {
                "use_onnxsim": False,
                "output_integer_quantized_tflite": True,
                "tflm_quant_type": "per_channel",
                "disable_group_convolution": False,
                "enable_batchmatmul_unfold": False,
                "input_layout": args.input_layout,
                "data_sample": data_sample,
                "input_names": input_names,
                "debug": args.debug})
            print("🛠️ Generating TFLM model...")
            model_tflm = onnx2tflm(model_onnx, tflm_args)
            if not model_tflm:
                print(f"❌ TFLM export failed")
                return None

        elif fmt == "vela":
            tflm_args = get_args(args, {
                "use_onnxsim": False,
                "output_integer_quantized_tflite": True,
                "tflm_quant_type": "per_channel",
                "disable_group_convolution": False,
                "enable_batchmatmul_unfold": False,
                "input_layout": args.input_layout,
                "data_sample": data_sample,
                "input_names": input_names,
                "debug": args.debug})
            model_tflm = onnx2tflm(model_onnx, tflm_args)
            if not model_tflm:
                print(f"❌ TFLM export failed")
                return None
            else:
                vela_args = get_args(args, {
                    "target_hardware": hw,
                    "config_vela": None,
                    "config_vela_system": "internal-default",
                    "force_symmetric_int_weights": False,
                    "memory_mode": "internal-default",
                    "tensor_allocator": "HillClimb",
                    "max_block_dependency": 3,
                    "arena_cache_size": None,
                    "cpu_tensor_alignment": 16,
                    "recursion_limit": 1000,
                    "hillclimb_max_iterations": 99999,
                    "vela_optimise": "Performance",
                    "model_name": args.model_name,
                    "bit_width": args.bit_width,
                    "debug": args.debug})
                out_name = model_onnx.split('.')[0]
                model_vela = vela.export(out_name, vela_args)
                if not model_vela:
                    print(f"❌ TFLM export failed")
                    return None
                else:
                    print(f"✅ VELA export success. Outputs saved to {model_ai8x}")
                    if hw == "hxwe2":
                        hxwe2_code_gen(model_vela, args.input_shape, sum([x * y for x, y in zip(args.output_shape, args.output_shape)]), args.overwrite)
                
        elif fmt == "ai8x":
            setup_ai8x()
            ai8x_args = get_args(args, {
                "avg_pool_rounding": False,
                "simple1b": False,
                "config_file": None,
                "prefix": args.out_dir,
                "test_dir": args.out_dir,
                "board_name": "EvKit_V1",
                "overwrite": args.overwrite,
                "compact_weights": False,
                "mlator": False,
                "softmax": False,
                "boost": None,
                "no_wfi": False,
                "zero_sram": False,
                "zero_unused": False,
                "max_speed": False,
                "fifo": False,
                "fast_fifo": False,
                "fast_fifo_quad": False,
                "riscv": False,
                "riscv_exclusive": False,
                "input_offset": None,
                "write_zero_registers": False,
                "no_unload": False,
                "no_deduplicate_weights": False,
                "no_scale_output": False,
                "qat_policy": None,
                "clip_method": None,
                "q_scale": "0.85"})
            model_ai8x = ai8x.export(model_ckpt, hw, data_sample, ai8x_args, args)
            if not model_ai8x:
                print(f"❌ AI8X export failed")
                return None
            else:
                print(f"✅ AI8X export success. Outputs saved to {model_ai8x}")

        elif fmt == "eiq":
            tflm_args = get_args(args, {
                "use_onnxsim": False,
                "output_integer_quantized_tflite": True,
                "tflm_quant_type": "per_channel",
                "disable_group_convolution": False,
                "enable_batchmatmul_unfold": False,
                "input_layout": args.input_layout,
                "data_sample": data_sample,
                "input_names": input_names,
                "debug": args.debug})
            model_tflm = onnx2tflm(model_onnx, tflm_args)
            if not model_tflm:
                print(f"❌ TFLM export failed")
                return None
            else:
                eiq_args = get_args(args, {
                    "eiq_path": "/opt/nxp/eIQ_Toolkit_v1.12.1/bin/neutron-converter/MCU_SDK_2.16.000/neutron-converter",
                    "out_dir": args.out_dir,
                    "debug": args.debug})
                model_eiq = eiq.export(model_tflm, hw, model_name, eiq_args)
                if not model_eiq:
                    print(f"❌ eIQ export failed")
                    return None
                else:
                    print(f"✅ eIQ export success. Outputs saved to {model_ai8x}")
                    if hw == "mcxn947":
                        mcxn947_code_gen(model_eiq, args.input_shape, sum([x * y for x, y in zip(args.output_shape, args.output_shape)]), args.overwrite)

        elif fmt == "cvi":
            cvi_args = get_args(args, {
                "bit_width": args.bit_width,
                "use_onnxsim": False,
                "output_integer_quantized_tflite": True,
                "tflm_quant_type": "per_channel",
                "disable_group_convolution": False,
                "enable_batchmatmul_unfold": False,
                "input_layout": args.input_layout,
                "data_sample": data_sample,
                "input_names": input_names,
                "input_shape": args.input_shape,
                "output_names": args.output_names,
                "target_hardware": hw,
                "calibration_table": None,
                "tolerance": 0.99,
                "dynamic": False,
                "debug": args.debug})
            model_cvi = cvi.export(model_onnx, data_sample, cvi_args)
            if not model_cvi:
                print(f"❌ CVI export failed")
                return None
            else:
                print(f"✅ CVI export success. Outputs saved to {model_ai8x}")
    return 1