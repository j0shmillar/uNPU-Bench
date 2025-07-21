from utils import run_subproc

import os
import sys

def export(base_name, target_hardware, model_name, eiq_args):
    tflm_model = f"{base_name}/{model_name}_full_integer_quant.tflite"
    eiq_model = f"{base_name}/{model_name}_full_integer_quant_eiq.tflite"

    pth = os.environ.get("EIQ_NEUTRON_PATH")
    if not pth:
        sys.exit("\033[91mNeutron compiler not found. Make sure EIQ_NEUTRON_PATH is set.\033[0m")

    eiq_cmd = [
        pth,
        "--input", tflm_model,
        "--output", eiq_model,
        # "--custom-options", f"target {target_hardware}"  # TODO 
    ]

    if run_subproc(eiq_cmd, eiq_args['debug'], "eIQ compiler failed.") is None:
        return None
    return eiq_model