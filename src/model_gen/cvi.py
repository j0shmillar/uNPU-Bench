from utils import run_subproc, gen_calibration_table

import os
import shutil
import numpy as np

def export(onnx_path, data_sample, args):
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    model_mlir = f"{os.path.splitext(onnx_path)[0]}.mlir"
    output_path = f"{os.path.splitext(onnx_path)[0]}.cvimodel"
    cal_table = args.get("calibration_table") or gen_calibration_table(".table")

    out_str = ", ".join(args["output_names"])

    transform_cmd = [
        "model_transform.py",
        "--model_name", model_name,
        "--model_def", onnx_path,
        "--mlir", model_mlir,
        "--output_names", out_str]

    data = np.load(data_sample)
    if data.ndim == 4:
        data = data[0]
    os.makedirs("cvi_data", exist_ok=True)
    temp_path = "cvi_data/ds_sample.npy"
    np.save(temp_path, data.astype(np.int64))

    optional_args = {
        "--model_data": args.get("caffe_model"),
        "--resize_dims": args.get("resize_dims"),
        "--pixel_format": args.get("pixel_format"),
        "--test_result": args.get("val_result_file"),
        "--excepts": args.get("excepts")}

    for k, v in optional_args.items():
        if v:
            transform_cmd += [k, v]
    if args.get("keep_aspect_ratio"):
        transform_cmd.append("--keep_aspect_ratio")

    print("Transforming...")

    if run_subproc(transform_cmd, args['debug'], "CVI - transform failed") is None:
        return None

    cali_cmd = [
        "run_calibration.py",
        "/workspace/src/" + model_mlir,
        "--dataset", "/workspace/src/cvi_data/",
        "--input_num", str(args["input_shape"][0]),
        "-o", cal_table]

    print("Calibrating...")

    if run_subproc(cali_cmd, args['debug'], "CVI - Calib failed") is None:
        return None

    quant = {4: 'INT4', 8: "INT8", 16: "F16", 32: "F32"}
    if args["bit_width"] in [8, 16, 32]:
        bit_width = quant[args["bit_width"]]
    elif args["bit_width"] in ['F32', 'BF16', 'F16', 'INT8', 'INT4', 'W8F16', 'W8BF16', 'W4F16', 'W4BF16', 'F8E4M3', 'F8E5M2', 'QDQ']:
        bit_width = args["bit_width"]
    else:
        print(f"Bit width {bit_width} not supported for CVI")
        return None
    
    deploy_cmd = [
        "model_deploy.py",
        "--mlir", model_mlir,
        "--quantize", quant[args["bit_width"]],
        "--calibration_table", cal_table,
        "--processor", args["target_hardware"],
        "--tolerance", str(args["tolerance"]),
        "--model", output_path]
    
    if args.get("dynamic"):
        deploy_cmd.append("--dynamic")
    if args.get("excepts"):
        deploy_cmd += ["--excepts", args["excepts"]]

    print("Deploying...")

    if run_subproc(deploy_cmd, args['debug'], "CVI - Compile failed") is None:
        return None

    shutil.rmtree("/workspace/src/cvi_data/")

    return output_path