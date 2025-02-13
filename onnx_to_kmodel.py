import os
import argparse
import numpy as np
import onnxsim
import onnx
import nncase

def parse_model_input_output(model_file):
    onnx_model = onnx.load(model_file)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    input_tensors = [node for node in onnx_model.graph.input if node.name in input_names]
    inputs = []
    onnx_type = input_tensors[0].type.tensor_type
    input_dict = {}
    input_dict['name'] = input_tensors[0].name
    input_dict['dtype'] = onnx.helper.tensor_dtype_to_np_dtype(onnx_type.elem_type)
    input_dict['shape'] = [(i.dim_value if i.dim_value != 0 else d) for i, d in zip(onnx_type.shape.dim, [1, 3, 32, 32])] #TODO dynamic shape
    inputs.append(input_dict)
    return onnx_model, inputs


def onnx_simplify(model_file, dump_dir):
    onnx_model, inputs = parse_model_input_output(model_file)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    input_shapes = {}
    for input in inputs:
        input_shapes[input['name']] = input['shape']
    onnx_model, check = onnxsim.simplify(onnx_model, overwrite_input_shapes=input_shapes)
    model_file = os.path.join(dump_dir, 'simplified.onnx')
    onnx.save_model(onnx_model, model_file)
    return model_file


def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content


def generate_data(shapes, sample):
    data_all = []
    for shape in shapes:
        data = []
        for i in range(sample):
            data.append(np.random.randint(0, 1, shape).astype(np.float32))
        data_all.append(data)
    return data_all

def main():
    parser = argparse.ArgumentParser(prog="nncase")
    #TODO add non random data option
    parser.add_argument("--target", type=str, default='k230', help='target to run')
    parser.add_argument("--model", type=str, default='avg_10.onnx', help='onnx model file')
    parser.add_argument("--kmodel", type=str, default='avg_10.kmodel', help='kmodel file')
    parser.add_argument("--ptq", type=str, help='ptq method,such as int8,int16,wint16,NoClip_int16,NoClip_wint16')
    args = parser.parse_args()

    input_shapes = [[1, 3, 32, 32]] #TODO make shape dynamic

    ptq_method = args.ptq

    dump_dir = 'tmp/nanotracker_head'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model_file = onnx_simplify(args.model, dump_dir)

    compile_options = nncase.CompileOptions()
    compile_options.target = args.target
    compile_options.preprocess = False
    compile_options.swapRB = True
    compile_options.input_type = 'float32'
    compile_options.input_range = [0, 255]
    compile_options.mean = [0, 0, 0]
    compile_options.std = [1, 1, 1]

    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = dump_dir

    if ptq_method=="int8" or ptq_method == "int16":
        compile_options.quant_type = ptq_method
    elif ptq_method == "wint16":
        compile_options.w_quant_type = 'int16'
    elif ptq_method == "NoClip_int16":
        compile_options.calibrate_method = 'NoClip'
        compile_options.quant_type = 'int16'
    elif ptq_method == "NoClip_wint16":
        compile_options.calibrate_method = 'NoClip'
        compile_options.w_quant_type = 'int16'
    else:
        pass

    compile_options.calibrate_method = 'NoClip'
    compile_options.quant_type = 'uint8'

    compiler = nncase.Compiler(compile_options)

    model_content = read_model_file(model_file)
    import_options = nncase.ImportOptions()
    compiler.import_onnx(model_content, import_options)

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.set_tensor_data(generate_data(input_shapes, ptq_options.samples_count))
   
    compiler.use_ptq(ptq_options)

    compiler.compile()

    kmodel = compiler.gencode_tobytes()
    with open(args.kmodel, 'wb') as f:
        f.write(kmodel)


if __name__ == '__main__':
    main()
