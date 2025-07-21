/*
 * Copyright 2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_s_microOpResolver.h"
#include "tensorflow/lite/micro/kernels/neutron/neutron.h"

tflite::MicroOpResolver &MODEL_GetOpsResolver()
{
    static tflite::MicroMutableOpResolver<18> s_microOpResolver;

    s_microOpResolver.AddPad();
    s_microOpResolver.AddConcatenation();
    s_microOpResolver.AddSlice();
    s_microOpResolver.AddResizeNearestNeighbor();
    s_microOpResolver.AddTranspose();
    s_microOpResolver.AddSplit();
    s_microOpResolver.AddConv2D();
    s_microOpResolver.AddRelu();
	s_microOpResolver.AddRelu6();
    s_microOpResolver.AddMul();
    s_microOpResolver.AddAdd();
    s_microOpResolver.AddSub();
    s_microOpResolver.AddMaxPool2D();
	s_microOpResolver.AddAveragePool2D();
    s_microOpResolver.AddDepthwiseConv2D();
    s_microOpResolver.AddReshape();
    s_microOpResolver.AddLogistic();
    s_microOpResolver.AddCustom(tflite::GetString_NEUTRON_GRAPH(), tflite::Register_NEUTRON_GRAPH());

    return s_microOpResolver;
}
