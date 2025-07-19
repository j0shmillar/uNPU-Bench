#!/bin/bash

python main.py \
    --model model/resnet/resnet.py \
    --model_ckpt model/resnet/resnet.pth.tar \
    --model_name ai85ressimplenet \
    --model_module_name AI85ResNet \
    --target_format ai8x \
    --target_hardware max78000 \
    --data_sample model/resnet/sample_data_nchw.npy \
    --input_shape 1 3 32 32 \
    --output_shape 100 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --avg_pool_rounding \
    --q_scale 0.85 \
    --config_file ai8x-synthesis/networks/cifar100-ressimplenet.yaml \
    --out_dir model/resnet/test \
    --overwrite