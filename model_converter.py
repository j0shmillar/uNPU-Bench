import os
import sys
import argparse
import numpy as np
import torch
import onnx
import onnx2tf
#from onnx_tf.backend import prepare

def convert_from_ckpt(input_file, args):
    #onnx_model = onnx.load(input_file)
    #tf_rep = prepare(onnx_model)
    #tf_rep.export_graph("tf_out")
    try:
        print("Converting ", input_file)
        onnx2tf.convert(
            input_onnx_file_path=args.model,
            output_folder_path="./test_out",
            not_use_onnxsim=True,
            verbosity="debug",  # Change to debug for more info
            output_integer_quantized_tflite=True,
            custom_input_op_name_np_data_path=[
                ["input", "./model/unet/sample_unet.npy", # ./model/yolo/sample_yolov1_rev_3.npy
                 np.random.rand(3).tolist(), 
                 np.random.rand(3).tolist()]
            ],
            quant_type="per-tensor",
            disable_group_convolution=True,
            enable_batchmatmul_unfold=True,
            # copy_onnx_input_output_names_to_tflite=True,
        )
        print("Exported to ./test_out")
        return 1
    except Exception as e:
        print(f"Conversion error: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='test_sbx_replaced.onnx', help='path to checkpoint file') 
    parser.add_argument('-q', '--quantize', action='store_true', default=True, help='quantize tflite model')
    parser.add_argument('-n', '--n_samples', default=10, help='number of rep samples for quantization')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbose mode')
    args = parser.parse_args()
    
    convert_from_ckpt(args.model, args)

if __name__ == '__main__':
    main()
