#!/bin/bash

python3 main.py \
    --model model/yolo/yolov1_96.py \
    --model_ckpt model/yolo/yolov1.pth.tar \
    --model_name ai85yolo96 \
    --model_module_name Yolov1_net \
    --target_format eiq \
    --target_hardware mcxn947 \
    --data_sample model/yolo/sample_data_nchw.npy \
    --input_shape 1 3 96 96 \
    --output_shape 10 12 2 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --out_dir model/yolo/out