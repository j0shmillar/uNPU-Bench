# python train.py --deterministic --model ai85unetsmall --out-fold-ratio 1 --dataset CamVid_s80_c3 --device MAX78000 --qat-policy None --use-bias --cpu --regression

import argparse
import sys

import onnx
import torch
import onnx2tf

import numpy as np

import unet_max78000 as mod

sys.path.append("/Users/joshmillar/Desktop/phd/mcu-nn-eval/ai8x-training") # TODO fix
import ai8x

ai8x.set_device(85, 0, True)

sys.path.append("/Users/joshmillar/Desktop/phd/mcu-nn-eval/") # TODO fix
import rm_unsupported_ops

args = None

def exp2_symbolic(g, input):
    log2 = g.op("Constant", value_t=torch.tensor(2.0).log())
    scaled_input = g.op("Mul", input, log2)
    return g.op("Exp", scaled_input) 

torch.onnx.register_custom_op_symbolic("aten::exp2", exp2_symbolic, opset_version=12)

def convert(input_file, arguments):

    print("converting ", input_file)
    checkpoint = torch.load(input_file, map_location='cpu', weights_only=False)

    if 'state_dict' not in checkpoint:
        print("no `state_dict` in file.")
        return 0

    checkpoint_state = checkpoint['state_dict']
    if arguments.verbose:
        print(f"\nmodel keys (state_dict):\n{', '.join(list(checkpoint_state.keys()))}")

    model = mod.AI85UNet(num_classes=3, num_channels=3, dimensions=(80, 80))

    model.load_state_dict(checkpoint_state)
    model.eval()

    input_fp32 = torch.randn(1, 3, 80, 80)

    onnx_f = input_file.replace('.pth.tar', '.onnx') # TODO fix

    try:
        torch.onnx.export(
            model,
            input_fp32,
            onnx_f,
            export_params=True,
            do_constant_folding=True, 
            input_names=['input'],
            output_names=['output'])
        print(f"ONNX model saved to {onnx_f}")
        onnx_model = onnx.load(onnx_f)
        print("exported model opset:", onnx_model.opset_import[0].version)
    except Exception as e:
        print(f"failed to export ONNX model: {e}")
        return 0
    
    input_file = onnx_f

    rm_unsupported_ops.run(input_file, input_file, unsupported_ops=[], placeholder_shape = [1, 1, 80, 80]) # TODO fix...class or regress?

    try:
        print("converting to tf/tflite...", input_file)
        onnx2tf.convert(
            input_onnx_file_path=input_file,
            output_folder_path=args.path,
            not_use_onnxsim=True,
            verbosity="debug", 
            output_integer_quantized_tflite=True,
            custom_input_op_name_np_data_path=[
                ["input", "./model/unet/sample_data_nhwc.npy", np.random.rand(3).tolist(), np.random.rand(3).tolist()]], # TODO fix
            quant_type="per-tensor",
            disable_group_convolution=True,
            enable_batchmatmul_unfold=True)
        print(f"exported to {args.path}")
        return 1
    except Exception as e:
        print(f"conversion error: {e}")
        return 0

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='./model/unet/unet.pth.tar', help='path to checkpoint file') 
    parser.add_argument('-q', '--quantize', action='store_true', default=False, help='quantize tflite model')
    parser.add_argument('-n', '--n_samples', default=10, help='number of rep samples for quantization')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbose mode')
    parser.add_argument('-p', '--path', default='./model/unet/quant', help='output folder path')
    args = parser.parse_args()

    convert(args.model, args)

if __name__=='__main__':
    main()

