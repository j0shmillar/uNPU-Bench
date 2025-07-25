#!/bin/bash

python3 main.py \
    --model model/nas/nas.py \
    --model_ckpt model/nas/nas.pth.tar \
    --model_name ai85nas \
    --model_module_name AI85NASCifarNet \
    --target_format cvi \
    --target_hardware cv180x \
    --data_sample model/nas/sample_data_nchw.npy \
    --input_shape 1 3 32 32 \
    --output_shape 10 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --out_dir model/nas/out