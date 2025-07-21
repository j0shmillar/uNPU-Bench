#!/bin/bash

python3 main.py \
    --model model/autoenc/autoenc.py \
    --model_ckpt model/autoenc/autoenc.pth.tar \
    --model_name ai85autoencoder \
    --model_module_name AI85AutoEncoder \
    --target_format vela \
    --target_hardware hxwe2 \
    --data_sample model/autoenc/sample_data_nhwc.npy \
    --input_layout NCW \
    --input_shape 1 256 3 \
    --output_shape 3 256 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --out_dir model/autoenc/out