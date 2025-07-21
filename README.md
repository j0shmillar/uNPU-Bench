# µNPU-Bench

This repository includes a model compiler wrapper and deployment pipeline for a variety of MCU-scale neural processing units (µNPUs). It automates model export, quantization, compilation, and deployment code generation using platform-specific toolchains — all from a single Torch-based source model.

---

## Supported Platforms & Formats  

| Format  | Target Hardware          | 
|---------|--------------------------|
| onnx    | All                      |
| tflm    | All                      |
| ai8x    | MAX78000, MAX78002       |
| vela    | Ethos-U55/U65 (incl. Himax WE2)     |
| eiq     | MCXN947 & other NXP NPUs     | 
| cvi     | CVITEK NPUs              |

---

## Features

- Fully declarative CLI: Compiler args are defined in platforms.yaml, not hardcoded.
- Automatically handles multi-stage dependencies. 
- Generates device-specific source trees for deployment.
- Automatically handles PTQ or QAT when needed.
- Preserves builds with backups unless --overwrite.

--- 
## Setup  

### Automatic: Docker (All toolchains)  
```bash
docker build -t unpu-bench .  
docker run --rm -it -v $(pwd):/workspace unpu-bench bash  
```
**⚠️ Requires Linux x86_64 (for CVI & eIQ support).**

### 🔧 Manual  
1. **Install Python dependencies:**  
   ```bash
        pip install -r requirements.txt  
    ```

2. **Optional: Platform-Specific Setup**   

| Platform             | Setup                                                                                                                                              |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **ai8x**             | - Clone repo: `git clone --recursive https://github.com/analogdevicesinc/ai8x-training`<br>- Set env var: `export AI8X_TRAIN_PATH=/path/to/ai8x-training` |
| **eIQ**              | - Download [eIQ Toolkit](https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT) <br>- Set env var: `export EIQ_NEUTRON_PATH=/path/to/neutron-converter` |
| **CVI**              | Use Docker image (Linux only)                                                                                                                       |
| **ONNX, TFLM, Vela** | No extra setup needed     

---

## Example Usage  
```bash
python3 main.py \
    --model model/yolo/yolov1_96.py \
    --model_ckpt model/yolo/yolov1.pth.tar \
    --model_name ai85yolo96 \
    --model_module_name Yolov1_net \
    --target_format ai8x \
    --target_hardware max78000 \
    --data_sample model/yolo/sample_data_nchw.npy \
    --input_shape 1 3 96 96 \
    --output_shape 10 12 2 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --avg_pool_rounding \
    --q_scale 0.85 \
    --fifo \
    --config_file model/yolo/ai85-yolo-96-hwc.yaml \
    --out_dir model/yolo/out \
    --overwrite
```
---

## Structure 

```
├── main.py                # Entry point
├── parse.py               # Argparse + compiler routing
├── utils.py               # Export, quant, subproc helpers
├── platforms.yaml         # Flag/hardware/format specs
├── model_gen/             # Exporters for target formats/platforms
├── code_gen.py            # C/C++ codegen for deployment
├── templates/             # C/C++ project templates
├── model/                 # Your models!
```

## CLI Arguments  

| Argument              | Type      | Default       | Description |  
|-----------------------|-----------|---------------|-------------|  
| `--target_format`     | str       | Required      | One or more formats: `onnx, tflm, ai8x, vela, eiq, cvi` |  
| `--target_hardware`   | str       | `max78000`    | Hardware target(s), e.g., `max78000, hxwe2, mcxn947, ethos-u55-64` |  
| `--model`             | path      | Required      | Python file containing model class |  
| `--model_ckpt`        | path      | Required      | Path to `.pth` checkpoint |  
| `--model_name`        | str       | Required      | Model base name (used for output files) |  
| `--model_module_name` | str       | Required      | Class name of the model in `--model` |  
| `--model_module_args` | str       | `None`        | Args passed to model constructor |  
| `--data_sample`       | `.npy`    | Required      | Representative input sample for quantization |  
| `--input_names`       | str       | Required      | Comma-separated names (e.g., `input`) |  
| `--input_shape`       | `list[int]` | Required    | Model input shape (e.g., `1 3 32 32`) |  
| `--input_layout`      | str       | `NCHW`        | Input layout: `NCHW, NHWC, NCW, NWC` |  
| `--output_names`      | str       | Required      | Comma-separated output names |  
| `--output_shape`      | `list[int]` | Required    | Model output shape |  
| `--bit_width`         | int       | `8`           | Quantization bit width (`4, 8, 16, 32`) |  
| `--out_dir`           | path      | Required      | Output directory for artifacts |  
| `--debug`             | bool      | `False`       | Print debug info from subprocesses |  
| `--overwrite`         | flag      | `False`       | Overwrite output directory if exists |  

### Platform-Specific Dynamic Arguments

Below are additional CLI arguments specific to each target format, defined in `platforms.yaml`.

#### `ai8x`

| Argument               | Type    | Default   | Description |
|------------------------|---------|-----------|-------------|
| `--qat_policy`         | str     | `None`    | Optional QAT policy name |
| `--clip_method`        | str     | `None`    | Clipping strategy (e.g. `avg`, `laplace`) |
| `--q_scale`            | float   | `0.85`    | PTQ scale factor |
| `--config_file`        | str     | `None`    | Path to synthesis config YAML |
| `--compact_weights`    | flag    | `False`   | Enable compact weight format |
| `--simple1b`           | flag    | `False`   | Simplify weights to 1-bit precision |
| `--no_wfi`             | flag    | `False`   | Disable Wait-for-Interrupts |
| `--fifo`               | flag    | `False`   | Enable FIFO path |
| `--max_speed`          | flag    | `False`   | Prioritize speed over size |
| `--zero_sram`          | flag    | `False`   | Zero SRAM before loading |
| ... *(see platforms.yaml for all options)* |

#### `tflm`

| Argument                    | Type    | Default      | Description |
|-----------------------------|---------|--------------|-------------|
| `--tflm_quant_type`         | str     | `per_channel`| Quantization mode (`per_tensor`, `per_channel`) |
| `--use_onnxsim`             | bool    | `False`      | Whether to simplify ONNX graph |
| `--disable_group_convolution`| bool   | `False`      | Disables group conv unfolding |
| `--enable_batchmatmul_unfold`| bool   | `False`      | Enables batchmatmul rewrite |

#### `vela`

| Argument                     | Type    | Default          | Description |
|------------------------------|---------|------------------|-------------|
| `--config_vela`              | str     | `None`           | Optional config file path |
| `--config_vela_system`       | str     | `internal-default`| System config name |
| `--force_symmetric_int_weights` | flag | `False`          | Enforce symmetric INT8 weights |
| `--memory_mode`              | str     | `internal-default`| Memory allocation mode |
| `--tensor_allocator`         | str     | `HillClimb`      | Allocator strategy |
| `--max_block_dependency`     | int     | `3`              | Max block dependency level |
| `--arena_cache_size`         | int     | `None`           | Optional arena cache size in bytes |
| `--cpu_tensor_alignment`     | int     | `16`             | CPU alignment in bytes |
| `--hillclimb_max_iterations` | int     | `99999`          | Max iterations for HillClimb |
| `--recursion_limit`          | int     | `1000`           | Python recursion limit |
| `--vela_optimise`            | str     | `Performance`    | Optimization goal |

#### `cvi`

| Argument             | Type    | Default | Description |
|----------------------|---------|---------|-------------|
| `--calibration_table`| str     | `None`  | Path to calibration table or auto-gen |
| `--tolerance`        | float   | `0.99`  | Accuracy threshold for deployment |
| `--dynamic`          | flag    | `False` | Enable dynamic inference shape support |
| `--pixel_format`     | str     | `None`  | Image format (e.g. RGB) |
| `--resize_dims`      | str     | `None`  | Resize before inference |
| `--excepts`          | str     | `None`  | Exception rules (layer skips, etc) |
| `--keep_aspect_ratio`| flag    | `False` | Maintain aspect ratio on resize |

---

## Adding New Formats or Platforms

### Add Format
Define its compiler backend: model_gen/<yourformat>.py with export(...)
Optionally, add C codegen: code_gen.py

### Add to platforms.yaml

For example,

```bash
vela:
  depends_on: [tflm]
  flags:
    memory_mode:
      type: str
      default: "internal-default"
      help: "Memory mode for Vela"
    force_symmetric_int_weights:
      type: bool
      default: false
      help: "Use symmetric weights"
```

Dynamic CLI flags are parsed automatically.

---

## Citation

If you use this or find it helpful, please consider citing our work:

```bash
@misc{unpu-bench,
      title={Benchmarking Ultra-Low-Power $\mu$NPUs}, 
      author={Josh Millar and Yushan Huang and Sarab Sethi and Hamed Haddadi and Anil Madhavapeddy},
      year={2025},
      eprint={2503.22567},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.22567}, 
}
```
