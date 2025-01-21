import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai8x-training"))

import argparse

import ai8x
import onnx
import torch
import onnx2tf
import torch.onnx
import numpy as np

# TODO hierarchy structure needs big reorg

ai8x.set_device(85, 1, True) # TODO make arg
args = None

def exp2_symbolic(g, input):
    log2 = g.op("Constant", value_t=torch.tensor(2.0).log())
    scaled_input = g.op("Mul", input, log2)
    return g.op("Exp", scaled_input) 

torch.onnx.register_custom_op_symbolic("aten::exp2", exp2_symbolic, opset_version=12)

# TODO move elsewhere (user should call model_converter functions)
def representative_dataset_gen():
    for _ in range(args.n_samples):
        input_data = np.random.rand(1, 48, 88, 88).astype(np.float32) # unet
        # input_data = (np.random.rand(1, 224, 224, 3) * 255).astype(np.float32)
        yield {'input_1': input_data} # TODO make input name an arg

def convert_from_ckpt(input_file, args):
    print("Converting ", input_file)

    # TODO: port code over from onnx2tf
    # TODO: also port over model_summaries from distiller
    onnx2tf.convert(
        input_onnx_file_path=args.model, # TODO make arg
        output_folder_path="./test_out", # TODO make arg
        not_use_onnxsim=True,
        verbosity="error",  # note INT8-FP16 activation bug https://github.com/ultralytics/ultralytics/issues/15873
        output_integer_quantized_tflite=args.quantize,
        custom_input_op_name_np_data_path=[["input.1", "./model/unet/sample_unet.npy", np.random.rand(48).tolist(), np.random.rand(48).tolist()]], # TODO replace with args (either from dataset or random and if random then size)
        quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
        disable_group_convolution=True,  # for end-to-end model compatibility
        enable_batchmatmul_unfold=True,  # for end-to-end model compatibility
    )
    print("Exported to ./test_out") # TODO make arg

    # TODO run e.g. xxd -i trained/ai85-camvid-unet-large-q.tflite trained/ai85-camvid-unet-large-q.c

    return 1

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='./model/unet/unet_qat.onnx', help='path to checkpoint file') # or ./trained/Yolov1_checkpoint-q.pth.tar
    parser.add_argument('-q', '--quantize', action='store_true', default=True, help='quantize tflite model')
    parser.add_argument('-n', '--n_samples', default=10, help='number of rep samples for quantization')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbose mode')
    args = parser.parse_args()

    convert_from_ckpt(args.model, args)

if __name__=='__main__':
    main()

