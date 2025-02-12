import sys
import numpy as np

from rknn.api import RKNN

def parse_arg():
    if len(sys.argv) < 5:
        print("Usage: python {} [onnx_model_path] [dataset_path] [output_rknn_path] [n_channels]".format(sys.argv[0]))
        exit(1)

    model_path = sys.argv[1]
    dataset_path= sys.argv[2]
    output_path = sys.argv[3]
    c = sys.argv[4]

    return model_path, dataset_path, output_path, c

if __name__ == '__main__':
    model_path, dataset_path, output_path, c = parse_arg()

    rknn = RKNN(verbose=False)

    mean_values, std_values = np.zeros([c]), np.ones([c])
    rknn.config(mean_values=mean_values.tolist(), std_values=std_values.tolist(), target_platform='rv1103', disable_rules=['convert_identity_to_reshape'])

    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('load model failed')
        exit(ret)
    print('done')

    ret = rknn.build(do_quantization=True, dataset=dataset_path) # false?
    if ret != 0:
        print('build model failed!')
        exit(ret)
    print('done')

    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
