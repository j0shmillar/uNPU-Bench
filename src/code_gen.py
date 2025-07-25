import os
import re
import shutil

import numpy as np

def replace_define(content, key, value):
    pattern = rf'^\s*#define\s+{key}(\s+\S*)?\s*$'
    replacement = f"#define {key} {value}"
    
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        content += f"\n{replacement}"
    
    return content

def patch_model_defines(filepath, model_config):
    with open(filepath, "r") as f:
        content = f.read()

    for key, value in model_config.items():
        content = replace_define(content, key, value)

    with open(filepath, "w") as f:
        f.write(content)

    print(f"Updated {filepath} with defines: {model_config}")

def generate_model_cc(tflite_path, output_cc_path, array_name="g_model_data", header_name="model_data.h"):
    with open(tflite_path, "rb") as f:
        model_data = f.read()

    hex_lines = []
    for i in range(0, len(model_data), 12):
        line = ", ".join(f"0x{b:02x}" for b in model_data[i:i+12])
        hex_lines.append("  " + line + ",")

    cc_code = f"""#include "{header_name}"

#include <cstdint>

alignas(16) const unsigned char {array_name}[] = {{
{os.linesep.join(hex_lines)}
}};
"""

    with open(output_cc_path, "w") as f:
        f.write(cc_code)

    print(f"C model written to {output_cc_path} with {len(model_data)} bytes.")
    return output_cc_path

def _prepare_template(src, dst, overwrite=False):
    if os.path.exists(dst):
        if overwrite:
            print(f"🗑️ Overwriting existing directory {dst}")
            shutil.rmtree(dst)
        else:
            base_backup = dst + "_backup"
            backup_path = base_backup
            i = 1
            while os.path.exists(backup_path):
                backup_path = f"{base_backup}_{i}"
                i += 1
            print(f"📁 Directory {dst} already exists. Moving it to {backup_path}")
            shutil.move(dst, backup_path)
            print(f"📦 Existing directory moved to {backup_path}")
    shutil.copytree(src, dst)
    print(f"Template copied from {src} to {dst}")

def mcxn947_code_gen(out_eiq, input_shape, output_shape_concat, overwrite):  
    src = "templates/mcxn947"
    out_dir = os.path.dirname(out_eiq)
    dst = os.path.join(out_dir, "mcxn947")

    _prepare_template(src, dst, overwrite)

    input_shape = np.array(input_shape).squeeze()

    if input_shape.ndim == 2:
        model_config = {
            'MODEL_IN_W': input_shape[1],
            'MODEL_IN_H': 1,
            'MODEL_IN_C': input_shape[0],
            'OUT_SIZE': output_shape_concat}
    else:
        model_config = {
            'MODEL_IN_W': input_shape[1],
            'MODEL_IN_H': input_shape[2],
            'MODEL_IN_C': input_shape[0],
            'OUT_SIZE': output_shape_concat}

    input_file = os.path.join(dst, "source", "infer.cpp")
    patch_model_defines(input_file, model_config)

    model_dst = os.path.join(dst, "source", "model", "model.tflite")
    os.makedirs(os.path.dirname(model_dst), exist_ok=True)
    shutil.copy(out_eiq, model_dst)

    print(f"Model inf template saved to {dst}")

def hxwe2_code_gen(out_vela, input_shape, output_shape_concat, overwrite):
    print(input_shape)
    src = "templates/hxwe2"
    out_dir = os.path.dirname(out_vela)
    dst = os.path.join(out_dir, "hxwe2")

    _prepare_template(src, dst, overwrite)

    input_shape = np.array(input_shape).squeeze()

    if input_shape.ndim == 2:
        model_config = {
            'MODEL_IN_W': input_shape[1],
            'MODEL_IN_H': 1,
            'MODEL_IN_C': input_shape[0],
            'OUT_SIZE': output_shape_concat}
    else:
        model_config = {
            'MODEL_IN_W': input_shape[1],
            'MODEL_IN_H': input_shape[2],
            'MODEL_IN_C': input_shape[0],
            'OUT_SIZE': output_shape_concat}

    input_file = os.path.join(dst, "app", "scenario_app", "template", "cvapp.cpp")
    patch_model_defines(input_file, model_config)

    output_cc_path = os.path.join(dst, "app", "scenario_app", "template", "model_data.cc")
    generate_model_cc(out_vela, output_cc_path)

    print(f"Model inf template saved to {dst}")