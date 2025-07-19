from model_gen import run_ai8x, run_vela, run_eiq, run_cvi
from code_gen import hxwe2_code_gen, mcxn947_code_gen
from pth_to_ckpt import pth_to_pth_tar
from utils import torch2onnx, onnx2tflm, setup_ai8x
from utils import make_out_dir
from itertools import product

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
                onnx_args = {
                    "opset": args.opset,
                    "opset_version": args.opset,
                    "custom_opsets": args.custom_opsets,
                    "export_params": args.export_params,
                    "do_constant_folding": args.do_constant_folding,
                    "input_names": input_names,
                    "input_shape": args.input_shape,
                    "output_names": output_names,
                    "keep_initializers_as_inputs": args.keep_initializers_as_inputs,
                    "dynamic_axes": args.dynamic_axes,
                    "model_name": args.model_name
                }
                model_onnx = torch2onnx(model, model_ckpt, onnx_args, args.out_dir)
                if not model_onnx:
                    print(f"❌ ONNX export failed for bit-width {args.bit_width}")
                    continue
                onnx_cache[args.bit_width] = model_onnx
            else:
                model_onnx = onnx_cache[args.bit_width]

        if fmt == "onnx":
            continue

        elif fmt == "tflm":
            tflm_args = {
                "use_onnxsim": args.use_onnxsim,
                "output_integer_quantized_tflite": args.output_integer_quantized_tflite,
                "tflm_quant_type": args.tflm_quant_type,
                "disable_group_convolution": args.disable_group_convolution,
                "enable_batchmatmul_unfold": args.enable_batchmatmul_unfold,
                "input_layout": args.input_layout,
                "data_sample": data_sample,
                "input_names": input_names
            }
            print("🛠️ Generating TFLM model...")
            onnx2tflm(model_onnx, tflm_args)

        elif fmt == "vela":
            tflm_args = {
                "use_onnxsim": args.use_onnxsim,
                "output_integer_quantized_tflite": args.output_integer_quantized_tflite,
                "tflm_quant_type": args.tflm_quant_type,
                "disable_group_convolution": args.disable_group_convolution,
                "enable_batchmatmul_unfold": args.enable_batchmatmul_unfold,
                "input_layout": args.input_layout,
                "data_sample": data_sample,
                "input_names": input_names
            }
            out_tflm = onnx2tflm(model_onnx, tflm_args)
            if out_tflm:
                vela_args = {
                    "target_hardware": hw,
                    "config_vela": args.config_vela,
                    "config_vela_system": args.config_vela_system,
                    "force_symmetric_int_weights": args.force_symmetric_int_weights,
                    "memory_mode": args.memory_mode,
                    "tensor_allocator": args.tensor_allocator,
                    "max_block_dependency": args.max_block_dependency,
                    "arena_cache_size": args.arena_cache_size,
                    "cpu_tensor_alignment": args.cpu_tensor_alignment,
                    "recursion_limit": args.recursion_limit,
                    "hillclimb_max_iterations": args.hillclimb_max_iterations,
                    "vela_optimise": args.vela_optimise,
                    "model_name": args.model_name,
                    "bit_width": args.bit_width
                }
                out_name = model_onnx.split('.')[0]
                out_vela = run_vela(out_name, vela_args)
                if out_vela and hw == "hxwe2":
                    hxwe2_code_gen(out_vela, args.input_shape, sum([x * y for x, y in zip(args.output_shape, args.output_shape)]), args.overwrite)

        elif fmt == "ai8x":
            setup_ai8x()
            ai8x_args = {
                "avg_pool_rounding": args.avg_pool_rounding,
                "simple1b": args.simple1b,
                "config_file": args.config_file,
                "prefix": args.out_dir,
                "test_dir": args.out_dir,
                "board_name": args.board_name,
                "overwrite": args.overwrite,
                "compact_weights": args.compact_weights,
                "mlator": args.mlator,
                "softmax": args.softmax,
                "boost": args.boost,
                "no_wfi": args.no_wfi,
                "zero_sram": args.zero_sram,
                "zero_unused": args.zero_unused,
                "max_speed": args.max_speed,
                "fifo": args.fifo,
                "fast_fifo": args.fast_fifo,
                "fast_fifo_quad": args.fast_fifo_quad,
                "riscv": args.riscv,
                "riscv_exclusive": args.riscv_exclusive,
                "input_offset": args.input_offset,
                "write_zero_registers": args.write_zero_registers,
                "no_unload": args.no_unload,
                "no_deduplicate_weights": args.no_deduplicate_weights,
                "no_scale_output": args.no_scale_output,
                "qat_policy": args.qat_policy,
                "clip_method": args.clip_method,
                "q_scale": args.q_scale
            }
            run_ai8x(model_ckpt, hw, data_sample, ai8x_args, args)

        elif fmt == "eiq":
            tflm_args = {
                "use_onnxsim": args.use_onnxsim,
                "output_integer_quantized_tflite": args.output_integer_quantized_tflite,
                "tflm_quant_type": args.tflm_quant_type,
                "disable_group_convolution": args.disable_group_convolution,
                "enable_batchmatmul_unfold": args.enable_batchmatmul_unfold,
                "input_layout": args.input_layout,
                "data_sample": data_sample,
                "input_names": input_names
            }
            tflm_base = onnx2tflm(model_onnx, tflm_args)
            if tflm_base:
                eiq_args = {
                    "eiq_path": args.eiq_path,
                    "out_dir": args.out_dir
                }
                out_eiq = run_eiq(tflm_base, hw, model_name, eiq_args)
                if out_eiq and hw == "mcxn947":
                    mcxn947_code_gen(out_eiq, args.input_shape, sum([x * y for x, y in zip(args.output_shape, args.output_shape)]), args.overwrite)

        elif fmt == "cvi":
            cvi_args = {
                "bit_width": args.bit_width,
                "use_onnxsim": args.use_onnxsim,
                "output_integer_quantized_tflite": args.output_integer_quantized_tflite,
                "tflm_quant_type": args.tflm_quant_type,
                "disable_group_convolution": args.disable_group_convolution,
                "enable_batchmatmul_unfold": args.enable_batchmatmul_unfold,
                "input_layout": args.input_layout,
                "data_sample": data_sample,
                "input_names": input_names,
                "input_shape": args.input_shape,
                "output_names": args.output_names,
                "target_hardware": hw,
                "calibration_table": args.calibration_table,
                "tolerance": args.tolerance,
                "correctness": args.correctness,
                "dynamic": args.dynamic
            }
            run_cvi(model_onnx, data_sample, cvi_args)
