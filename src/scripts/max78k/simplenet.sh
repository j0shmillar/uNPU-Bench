#!/bin/bash

python main.py \
    --model model/simplenet/simplenet.py \
    --model_ckpt model/simplenet/simplenet.pth.tar \
    --model_name ai85simplenet \
    --model_module_name AI85SimpleNet \
    --target_format ai8x \
    --target_hardware max78000 \
    --data_sample model/simplenet/sample_data_nchw.npy \
    --input_shape 1 3 32 32 \
    --output_shape 100 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --avg_pool_rounding \
    --q_scale 0.85 \
    --config_file ai8x-synthesis/networks/cifar100-simple.yaml \
    --out_dir model/simplenet/test \
    --overwrite