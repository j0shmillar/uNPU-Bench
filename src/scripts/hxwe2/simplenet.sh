#!/bin/bash

python3 main.py \
    --model model/simplenet/simplenet.py \
    --model_ckpt model/simplenet/simplenet.pth.tar \
    --model_name ai85simplenet \
    --model_module_name AI85SimpleNet \
    --target_format vela \
    --target_hardware hxwe2 \
    --data_sample model/simplenet/sample_data_nchw.npy \
    --input_shape 1 3 32 32 \
    --output_shape 100 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --out_dir model/simplenet/out