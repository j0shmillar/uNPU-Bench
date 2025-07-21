#!/bin/bash

python3 main.py \
    --model model/autoenc/autoenc.py \
    --model_ckpt model/autoenc/autoenc.pth.tar \
    --model_name ai85autoencoder \
    --model_module_name AI85AutoEncoder \
    --target_format ai8x \
    --target_hardware max78000 \
    --data_sample model/autoenc/sample_data_norm.npy \
    --input_shape 256 3 \
    --output_shape 3 256 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --avg_pool_rounding \
    --q_scale 0.85 \
    --config_file ai8x-synthesis/networks/ai85-autoencoder.yaml \
    --out_dir model/autoenc/out \
    --overwrite