/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue Feb 18 16:03:25 2025
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "2d20351c09010923a1fc1bcc47a55495"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Tue Feb 18 16:03:25 2025"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24576, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  dense_3_dense_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  dense_3_dense_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  dense_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  dense_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 4, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  dense_5_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  dense_5_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  dense_6_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3072, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  dense_6_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 96, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  dense_7_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  dense_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 768, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  dense_3_dense_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 64, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  dense_4_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 32, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  dense_5_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 4, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  dense_6_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 32, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  dense_7_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 96, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 768, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  dense_3_dense_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  dense_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  dense_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  dense_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  dense_7_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 768, AI_STATIC)
/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002471465850248933f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010726391337811947f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_3_dense_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006527129560709f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_4_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014780465513467789f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_5_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004577660467475653f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_6_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004697443451732397f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_7_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017237815773114562f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_input0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05147700756788254f),
    AI_PACK_INTQ_ZP(27)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04686088487505913f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025124309584498405f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_3_dense_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016207361593842506f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014586389064788818f),
    AI_PACK_INTQ_ZP(-27)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_5_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008315013721585274f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_6_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005360534880310297f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005738004110753536f),
    AI_PACK_INTQ_ZP(67)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 128), AI_STRIDE_INIT(4, 1, 256, 256, 256),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 128, 3, 1, 64), AI_STRIDE_INIT(4, 1, 128, 384, 384),
  1, &conv2d_2_weights_array, &conv2d_2_weights_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_2_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  dense_3_dense_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 64, 32, 1, 1), AI_STRIDE_INIT(4, 1, 64, 2048, 2048),
  1, &dense_3_dense_weights_array, &dense_3_dense_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  dense_3_dense_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &dense_3_dense_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  dense_4_weights, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 32, 4, 1, 1), AI_STRIDE_INIT(4, 1, 32, 128, 128),
  1, &dense_4_weights_array, &dense_4_weights_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  dense_4_bias, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &dense_4_bias_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  dense_5_weights, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 4, 32, 1, 1), AI_STRIDE_INIT(4, 1, 4, 128, 128),
  1, &dense_5_weights_array, &dense_5_weights_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  dense_5_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &dense_5_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  dense_6_weights, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 32, 96, 1, 1), AI_STRIDE_INIT(4, 1, 32, 3072, 3072),
  1, &dense_6_weights_array, &dense_6_weights_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  dense_6_bias, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &dense_6_bias_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  dense_7_weights, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 96, 768, 1, 1), AI_STRIDE_INIT(4, 1, 96, 73728, 73728),
  1, &dense_7_weights_array, &dense_7_weights_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  dense_7_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &dense_7_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 6912, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6912, 6912),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  dense_3_dense_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 2, 2, 128, 128),
  1, &dense_3_dense_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  dense_4_scratch0, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 2, 2, 64, 64),
  1, &dense_4_scratch0_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  dense_5_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 2, 2, 8, 8),
  1, &dense_5_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  dense_6_scratch0, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 2, 2, 64, 64),
  1, &dense_6_scratch0_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  dense_7_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 2, 2, 192, 192),
  1, &dense_7_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input0_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 3), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &serving_default_input0_output_array, &serving_default_input0_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input0_output0, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 3, 1), AI_STRIDE_INIT(4, 1, 1, 256, 768),
  1, &serving_default_input0_output_array, &serving_default_input0_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 3, 1), AI_STRIDE_INIT(4, 1, 1, 128, 384),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conv2d_2_output_array, &conv2d_2_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  dense_3_dense_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &dense_3_dense_output_array, &dense_3_dense_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  dense_4_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 1, 1, 4, 4),
  1, &dense_4_output_array, &dense_4_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  dense_5_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &dense_5_output_array, &dense_5_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  dense_6_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &dense_6_output_array, &dense_6_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  dense_7_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 1, 1, 768, 768),
  1, &dense_7_output_array, &dense_7_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_7_weights, &dense_7_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_7_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_7_layer, 7,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_7_chain,
  NULL, &dense_7_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_6_weights, &dense_6_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_6_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_6_layer, 6,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_6_chain,
  NULL, &dense_7_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_5_weights, &dense_5_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_5_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_5_layer, 5,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_5_chain,
  NULL, &dense_6_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_3_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_4_weights, &dense_4_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_4_layer, 4,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_4_chain,
  NULL, &dense_5_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_3_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_3_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_3_dense_weights, &dense_3_dense_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_3_dense_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_3_dense_layer, 3,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_3_dense_chain,
  NULL, &dense_4_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_2_chain,
  NULL, &dense_3_dense_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_1_chain,
  NULL, &conv2d_2_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 140944, 1, 1),
    140944, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 7360, 1, 1),
    7360, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &dense_7_output),
  &conv2d_1_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 140944, 1, 1),
      140944, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 7360, 1, 1),
      7360, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &dense_7_output),
  &conv2d_1_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_input0_output_array.data = AI_PTR(g_network_activations_map[0] + 5120);
    serving_default_input0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5120);
    
    conv2d_1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 5888);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 5888);
    
    conv2d_1_output_array.data = AI_PTR(g_network_activations_map[0] + 6912);
    conv2d_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6912);
    
    conv2d_2_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_2_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_2_output_array.data = AI_PTR(g_network_activations_map[0] + 7296);
    conv2d_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7296);
    
    dense_3_dense_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_3_dense_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_3_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 128);
    dense_3_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 128);
    
    dense_4_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_4_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_4_output_array.data = AI_PTR(g_network_activations_map[0] + 64);
    dense_4_output_array.data_start = AI_PTR(g_network_activations_map[0] + 64);
    
    dense_5_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_5_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_5_output_array.data = AI_PTR(g_network_activations_map[0] + 8);
    dense_5_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8);
    
    dense_6_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 40);
    dense_6_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 40);
    
    dense_6_output_array.data = AI_PTR(g_network_activations_map[0] + 104);
    dense_6_output_array.data_start = AI_PTR(g_network_activations_map[0] + 104);
    
    dense_7_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 200);
    dense_7_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    
    dense_7_output_array.data = AI_PTR(g_network_activations_map[0] + 392);
    dense_7_output_array.data_start = AI_PTR(g_network_activations_map[0] + 392);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_network_weights_map[0] + 32768);
    conv2d_1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 32768);
    
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(g_network_weights_map[0] + 33280);
    conv2d_2_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 33280);
    
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(g_network_weights_map[0] + 57856);
    conv2d_2_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 57856);
    
    dense_3_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_3_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 58112);
    dense_3_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 58112);
    
    dense_3_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_3_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 60160);
    dense_3_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 60160);
    
    dense_4_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_4_weights_array.data = AI_PTR(g_network_weights_map[0] + 60288);
    dense_4_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 60288);
    
    dense_4_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_4_bias_array.data = AI_PTR(g_network_weights_map[0] + 60416);
    dense_4_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 60416);
    
    dense_5_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_5_weights_array.data = AI_PTR(g_network_weights_map[0] + 60432);
    dense_5_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 60432);
    
    dense_5_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_5_bias_array.data = AI_PTR(g_network_weights_map[0] + 60560);
    dense_5_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 60560);
    
    dense_6_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_6_weights_array.data = AI_PTR(g_network_weights_map[0] + 60688);
    dense_6_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 60688);
    
    dense_6_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_6_bias_array.data = AI_PTR(g_network_weights_map[0] + 63760);
    dense_6_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 63760);
    
    dense_7_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_7_weights_array.data = AI_PTR(g_network_weights_map[0] + 64144);
    dense_7_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 64144);
    
    dense_7_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_7_bias_array.data = AI_PTR(g_network_weights_map[0] + 137872);
    dense_7_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 137872);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 203108,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 203108,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_network_data_params_get(&params) != true) {
        err = ai_network_get_error(*network);
        return err;
    }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_network_init(*network, &params) != true) {
        err = ai_network_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

