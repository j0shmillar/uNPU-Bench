from utils import run_subproc

import shutil
import re
import os

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

    print(f"✅ Updated {filepath} with defines: {model_config}")

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

    print(f"✅ C source written to {output_cc_path} with {len(model_data)} bytes.")
    return output_cc_path

def _prepare_template(src, dst):
    if os.path.exists(dst):
        print(f"Directory {dst} already exists. Renaming it to avoid overwrite.")
        backup_path = dst + "_backup"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.move(dst, backup_path)
        print(f"📦 Existing directory moved to {backup_path}")
    shutil.copytree(src, dst)

def mcxn947_code_gen(out_eiq, input_shape, output_shape_concat):  
    src = "templates/mcxn947"
    out_dir = os.path.dirname(out_eiq)
    dst = os.path.join(out_dir, "mcxn947")

    _prepare_template(src, dst)

    model_config = {
        'MODEL_IN_W': input_shape[1],
        'MODEL_IN_H': input_shape[2],
        'MODEL_IN_C': input_shape[0],
        'OUT_SIZE': output_shape_concat
    }

    input_file = os.path.join(dst, "source", "infer.cpp")
    patch_model_defines(input_file, model_config)

    model_dst = os.path.join(dst, "source", "model", "model.tflite")
    os.makedirs(os.path.dirname(model_dst), exist_ok=True)
    shutil.copy(out_eiq, model_dst)

    print(f"✅ mcxn947 model inference code saved to {dst}.")

def hxwe2_code_gen(out_vela, input_shape, output_shape_concat):
    src = "templates/hxwe2"
    out_dir = os.path.dirname(out_vela)
    dst = os.path.join(out_dir, "hxwe2")

    _prepare_template(src, dst)

    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    model_config = {
        'MODEL_IN_W': input_shape[1],
        'MODEL_IN_H': input_shape[2],
        'MODEL_IN_C': input_shape[0],
        'OUT_SIZE': output_shape_concat
    }

    input_file = os.path.join(dst, "app", "scenario_app", "template", "cvapp.cpp")
    patch_model_defines(input_file, model_config)

    output_cc_path = os.path.join(dst, "app", "scenario_app", "template", "model_data.cc")
    generate_model_cc(out_vela, output_cc_path)

    print(f"✅ hxwe2 model inference code saved to {dst}.")