#!/bin/bash

python main.py \
    --model model/yolo/yolov1_96_max78000.py \
    --model_ckpt model/yolo/yolov1.pth.tar \
    --model_name ai85yolo96 \
    --model_module_name Yolov1_net \
    --target_format ai8x \
    --target_hardware max78000 \
    --data_samples model/yolo/sample_data_nhwc.npy \
    --input_shape 96 96 3 \
    --input_names input \
    --output_names output \
    --bit_width 8 \
    --avg_pool_rounding \
    --q_scale 0.85 \
    --fifo \
    --config_file model/yolo/ai85-yolo-96-hwc.yaml \
    --prefix yolo_test \
    --test_dir model/yolo/test

