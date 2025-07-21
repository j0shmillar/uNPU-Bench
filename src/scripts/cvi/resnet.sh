#!/bin/bash

python3 main.py \
    --model model/resnet/resnet.py \
    --model_ckpt model/resnet/resnet.pth.tar \
    --model_name resnet \
    --model_module_name AI85ResNet \
    --target_format cvi \
    --target_hardware cv180x \
    --data_sample model/resnet/sample_data_nchw.npy \
    --input_shape 1 3 32 32 \
    --output_shape 100 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --out_dir model/resnet/out


