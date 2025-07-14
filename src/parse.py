import numpy as np
from compile import run_ai8x, run_vela, run_eiq, run_cvi
from pth_to_ckpt import pth_to_pth_tar
from utils import torch2onnx, onnx2tflm, setup_ai8x

def compile(model, model_ckpt, target_formats, target_hardware, data_samples,
            input_shape, input_layout, input_names, output_names, args):
    print(f"data samples = {data_samples}")
    if model_ckpt.endswith(".pth"):
        model_ckpt_tar = model_ckpt + ".tar"
        pth_to_pth_tar(model_ckpt, model_ckpt_tar)
        model_ckpt = model_ckpt_tar

    success = False

    if "onnx" in target_formats:
        onnx_args = {
            "opset": args.opset,
            "opset_version": args.opset_version,
            "custom_opsets": args.custom_opsets,
            "export_params": args.export_params,
            "do_constant_folding": args.do_constant_folding,
            "input_names": input_names,
            "input_shape": args.input_shape,
            "output_names": output_names,
            "keep_initializers_as_inputs": args.keep_initializers_as_inputs,
            "dynamic_axes": args.dynamic_axe}
        model_onnx = torch2onnx(model, model_ckpt, onnx_args)
        success = success or (model_onnx is not None)

    if "tflm" in target_formats:
        model_onnx = torch2onnx(model, model_ckpt, args)
        if model_onnx:
            tflm_args = {
                "use_onnxsim": args.use_onnxsim,
                "output_integer_quantized_tflite": args.output_integer_quantized_tflite,
                "tflm_quant_type": args.tflm_quant_type,
                "disable_group_convolution": args.disable_group_convolution,
                "enable_batchmatmul_unfold": args.enable_batchmatmul_unfold,
                "input_layout": args.input_layout,
                "data_samples": data_samples,
                "input_names": input_names}
            print("\nGenerating TFLM model...")
            out_tflm = onnx2tflm(model_onnx, tflm_args)
            success = success or (out_tflm is not None)

    if "ai8x" in target_formats:
        setup_ai8x()
        ai8x_args = {
            "avg_pool_rounding": args.avg_pool_rounding,
            "simple1b": args.simple1b,
            "config_file": args.config_file,
            "prefix": args.prefix,
            "test_dir": args.test_dir,
            "board_name": args.board_name,
            "overwrite": args.overwrite,
            "compact_weights": args.compact_weights,
            "mlator": args.mlator,
            "softmax": args.softmax,
            "boost": args.boost,
            "timer": args.timer,
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
            "q_scale": args.q_scale}
        print("\nGenerating AI8X model and code...")
        out_ai8x = run_ai8x(model_ckpt, target_hardware, data_samples, args, ai8x_args)
        success = success or (out_ai8x is not None)

    if "vela" in target_formats:
        onnx_args = {
            "opset": args.opset,
            "opset_version": args.opset_version,
            "custom_opsets": args.custom_opsets,
            "export_params": args.export_params,
            "do_constant_folding": args.do_constant_folding,
            "input_names": input_names,
            "input_shape": args.input_shape,
            "output_names": output_names,
            "keep_initializers_as_inputs": args.keep_initializers_as_inputs,
            "dynamic_axes": args.dynamic_axe}
        model_onnx = torch2onnx(model, model_ckpt, onnx_args) 
        if model_onnx:
            vela_args = {
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
                "model_name": args.model_name}
            print("\nGenerating Vela model...")
            out_vela = run_vela(model_onnx, vela_args)
            success = success or (out_vela is not None)

    if "eiq" in target_formats:
        onnx_args = {
            "opset": args.opset,
            "opset_version": args.opset_version,
            "custom_opsets": args.custom_opsets,
            "export_params": args.export_params,
            "do_constant_folding": args.do_constant_folding,
            "input_names": input_names,
            "input_shape": args.input_shape,
            "output_names": output_names,
            "keep_initializers_as_inputs": args.keep_initializers_as_inputs,
            "dynamic_axes": args.dynamic_axe}
        model_onnx = torch2onnx(model, model_ckpt, onnx_args)
        if model_onnx:
            tflm_args = {
                "use_onnxsim": args.use_onnxsim,
                "output_integer_quantized_tflite": args.output_integer_quantized_tflite,
                "tflm_quant_type": args.tflm_quant_type,
                "disable_group_convolution": args.disable_group_convolution,
                "enable_batchmatmul_unfold": args.enable_batchmatmul_unfold,
                "input_layout": args.input_layout,
                "data_samples": data_samples,
                "input_names": input_names
            }
            print("\nGenerating TFLM model...")
            model_tflm = onnx2tflm(model_onnx, tflm_args)
            if model_tflm:
                eiq_args = {
                    "eiq_path": args.eiq_path
                }
                print("\nGenerating Vela model and code...")
                out_eiq = run_eiq(model_tflm, target_hardware, eiq_args)
                success = success or (out_eiq is not None)
    
    if 'cvi' in target_formats:
        onnx_args = {
            "opset": args.opset,
            "opset_version": args.opset_version,
            "custom_opsets": args.custom_opsets,
            "export_params": args.export_params,
            "do_constant_folding": args.do_constant_folding,
            "input_names": input_names,
            "input_shape": args.input_shape,
            "output_names": output_names,
            "keep_initializers_as_inputs": args.keep_initializers_as_inputs,
            "dynamic_axes": args.dynamic_axe}
        model_onnx = torch2onnx(model, model_ckpt, onnx_args)
        if model_onnx:     
            cvi_args = {
                "use_onnxsim": args.use_onnxsim,
                "output_integer_quantized_tflite": args.output_integer_quantized_tflite,
                "tflm_quant_type": args.tflm_quant_type,
                "disable_group_convolution": args.disable_group_convolution,
                "enable_batchmatmul_unfold": args.enable_batchmatmul_unfold,
                "input_layout": args.input_layout,
                "data_samples": data_samples,
                "input_names": input_names
            }
            print("\nGenerating CVI model and code...")   
            out_cvi = run_cvi(model_onnx, cvi_args)
            success = success or (out_cvi is not None)
    return success
