SUPPORTED_BIT_WIDTHS = {
    "ai8x": [1, 2, 4, 8],
    "vela": [8, 16, 32],
    "eiq": [8],
    "tflm": [8, 16, 32],
    "cvi": [8, 16, 32],
    "onnx": [32],
}

SUPPORTED_HARDWARE = [
    "max78000", "max78002", 
    "ethos-u55-32", "ethos-u55-64", "ethos-u55-128", "ethos-u55-256", "ethos-u65-256", "ethos-u65-512",
    "bm1684x",
    "mcxn947"
]

GLOBAL_FLAGS = {
    "model", "model_ckpt", "model_name", "model_module_name",
    "input_shape", "target_hardware", "target_format",
    "data_samples", "input_names", "output_names", "input_layout",
    "bit_width", "model_module_args"
}

PLATFORM_FLAGS = {
    "max78000": {
        "flags": [
            "avg_pool_rounding", "simple1b", "config_file", "display_checkpoint",
            "prefix", "test_dir", "board_name", "overwrite", "compact_weights",
            "mexpress", "no_mexpress", "mlator", "softmax", "boost", "timer",
            "no_wfi", "define", "no_pipeline", "max_speed", "fifo", "fast_fifo",
            "fast_fifo_quad", "riscv", "riscv_exclusive",
            "input_offset", "write_zero_registers", "init_tram", "zero_sram", "zero_unused", "max_verify_length",
            "no_unload", "no_deduplicate_weights", "no_scale_output",
            "qat_policy", "clip_method", "q_scale"
        ]
    },
    "vela": {
        "flags": [
            "config_vela", "config_vela_system", "force_symmetric_int_weights",
            "memory_mode", "tensor_allocator", "max_block_dependency",
            "arena_cache_size", "cpu_tensor_alignment", "recursion_limit",
            "hillclimb_max_iterations", "vela_optimise"
        ]
    },
    "tflm": {
        "flags": [
            "use_onnxsim", "output_integer_quantized_tflite",
            "tflm_quant_type", "disable_group_convolution",
            "enable_batchmatmul_unfold"
        ]
    },
    "onnx": {
        "flags": ["opset", "export_params", "do_constant_folding", "input_names", "output_names", 
                  "opset_version", "dynamic_axes", "keep_initializers_as_inputs", "custom_opsets"] 
    },
    "eiq": {
        "flags": ["eiq_path"]  # add if specific flags needed for ONNX
    },
    "cvi": {
        "flags": ["model_def", "debug", "quantize", "calibration_table",
                "tolerance", "correctness", "dynamic",]
    }
}