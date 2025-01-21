/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Fri Jan 17 15:43:13 2025
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
#define AI_NETWORK_MODEL_SIGNATURE     "172fded3d78c2eb2c925ec077923631a"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Fri Jan 17 15:43:13 2025"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7936, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9984, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_44_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7936, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_50_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  dense_57_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 512, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_x_10_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3072, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conversion_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3073, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conversion_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  nl_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conversion_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  nl_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conversion_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  nl_8_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conversion_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  nl_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conversion_12_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  nl_13_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conversion_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  nl_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  pool_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conversion_18_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  nl_19_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conversion_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  nl_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)
/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conversion_23_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)
/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  nl_24_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)
/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  conversion_25_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)
/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  nl_26_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)
/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  pool_27_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)
/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conversion_29_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)
/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  nl_30_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)
/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  conversion_31_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  nl_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  conversion_34_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)
/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  nl_35_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)
/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  conversion_36_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  nl_37_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  pool_38_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  conversion_40_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)
/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  nl_41_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)
/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  conversion_42_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  nl_43_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_44_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  conversion_45_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)
/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  nl_46_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)
/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  conversion_47_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  nl_48_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  pool_49_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)
/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_50_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)
/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  conversion_51_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)
/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  nl_52_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)
/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  conversion_53_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)
/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  nl_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)
/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  transpose_55_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)
/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  dense_57_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)
/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  conversion_58_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 100, AI_STATIC)
/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1728, AI_STATIC)
/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)
/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)
/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)
/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)
/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)
/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)
/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)
/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)
/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)
/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)
/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)
/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)
/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)
/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_44_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)
/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_44_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)
/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_50_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)
/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_50_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)
/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  dense_57_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)
/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  dense_57_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 100, AI_STATIC)
/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3820, AI_STATIC)
/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)
/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6528, AI_STATIC)
/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7552, AI_STATIC)
/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01865844801068306f),
    AI_PACK_INTQ_ZP(-14)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(14.239572525024414f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(14.239215850830078f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_5_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.933333396911621f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_6_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2183.65283203125f),
    AI_PACK_INTQ_ZP(-18)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_9_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2183.65478515625f),
    AI_PACK_INTQ_ZP(-18)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1239.37255859375f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(291487.375f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(291487.375f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_15_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(126532.8125f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_16_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(126532.8125f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_17_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(29468616.0f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_20_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(29468616.0f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_21_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(15125518.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_22_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3699321088.0f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_25_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3699321088.0f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_26_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1610501760.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_27_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1610501760.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_28_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(368863936512.0f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_31_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(368863936512.0f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_32_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(164659462144.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_33_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(55049103343616.0f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_36_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(55049103343616.0f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_37_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(25928535113728.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_38_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(25928535113728.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_39_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6705185367785472.0f),
    AI_PACK_INTQ_ZP(35)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_42_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6705185367785472.0f),
    AI_PACK_INTQ_ZP(35)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_43_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2417220210655232.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_44_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.681642795288494e+17f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_47_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.681642795288494e+17f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_48_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.7751102956476826e+17f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_49_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.7751102956476826e+17f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_50_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.667153401620516e+19f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_53_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.667153401620516e+19f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_54_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.257283514078908e+19f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_55_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.257283514078908e+19f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_57_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.449472214529018e+22f),
    AI_PACK_INTQ_ZP(91)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_6_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5039370059967041f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5039370059967041f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_17_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12598425149917603f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_22_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0078740119934082f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_28_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12598425149917603f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_33_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5039370059967041f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_39_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12598425149917603f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_44_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.25196850299835205f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_50_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5039370059967041f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_57_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0078740119934082f),
    AI_PACK_INTQ_ZP(0)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_scratch0, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_22_scratch0_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_scratch0, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 7936, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7936, 7936),
  1, &conv2d_28_scratch0_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_33_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_scratch0, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 9984, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9984, 9984),
  1, &conv2d_39_scratch0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_44_scratch0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 7936, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7936, 7936),
  1, &conv2d_44_scratch0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_50_scratch0, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_50_scratch0_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  dense_57_scratch0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 2, 2, 1024, 1024),
  1, &dense_57_scratch0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_x_10_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 32, 32), AI_STRIDE_INIT(4, 4, 4, 12, 384),
  1, &serving_default_x_10_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conversion_0_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 32, 32), AI_STRIDE_INIT(4, 1, 1, 3, 96),
  1, &conversion_0_output_array, &conversion_0_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conversion_2_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &conversion_2_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  nl_3_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &nl_3_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conversion_4_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &conversion_4_output_array, &conversion_4_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  nl_5_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &nl_5_output_array, &nl_5_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 32), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &conv2d_6_output_array, &conv2d_6_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conversion_7_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 32), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &conversion_7_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  nl_8_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 32), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &nl_8_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conversion_9_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 32), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &conversion_9_output_array, &conversion_9_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  nl_10_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 32), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &nl_10_output_array, &nl_10_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &conv2d_11_output_array, &conv2d_11_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conversion_12_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &conversion_12_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  nl_13_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &nl_13_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conversion_14_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &conversion_14_output_array, &conversion_14_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  nl_15_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 32), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &nl_15_output_array, &nl_15_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  pool_16_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 16), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &pool_16_output_array, &pool_16_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 1, 1, 32, 512),
  1, &conv2d_17_output_array, &conv2d_17_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conversion_18_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 4, 4, 128, 2048),
  1, &conversion_18_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  nl_19_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 4, 4, 128, 2048),
  1, &nl_19_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conversion_20_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 1, 1, 32, 512),
  1, &conversion_20_output_array, &conversion_20_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  nl_21_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 1, 1, 32, 512),
  1, &nl_21_output_array, &nl_21_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 16), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &conv2d_22_output_array, &conv2d_22_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conversion_23_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 16, 16), AI_STRIDE_INIT(4, 4, 4, 256, 4096),
  1, &conversion_23_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  nl_24_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 16, 16), AI_STRIDE_INIT(4, 4, 4, 256, 4096),
  1, &nl_24_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conversion_25_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 16), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &conversion_25_output_array, &conversion_25_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  nl_26_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 16), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &nl_26_output_array, &nl_26_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  pool_27_output, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 8, 8), AI_STRIDE_INIT(4, 1, 1, 64, 512),
  1, &pool_27_output_array, &pool_27_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 1, 1, 128, 1024),
  1, &conv2d_28_output_array, &conv2d_28_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conversion_29_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 4, 4, 512, 4096),
  1, &conversion_29_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  nl_30_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 4, 4, 512, 4096),
  1, &nl_30_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conversion_31_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 1, 1, 128, 1024),
  1, &conversion_31_output_array, &conversion_31_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  nl_32_output, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 1, 1, 128, 1024),
  1, &nl_32_output_array, &nl_32_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 1, 1, 128, 1024),
  1, &conv2d_33_output_array, &conv2d_33_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conversion_34_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 4, 4, 512, 4096),
  1, &conversion_34_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  nl_35_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 4, 4, 512, 4096),
  1, &nl_35_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conversion_36_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 1, 1, 128, 1024),
  1, &conversion_36_output_array, &conversion_36_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  nl_37_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 1, 1, 128, 1024),
  1, &nl_37_output_array, &nl_37_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  pool_38_output, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 4, 4), AI_STRIDE_INIT(4, 1, 1, 128, 512),
  1, &pool_38_output_array, &pool_38_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_output, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 4), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &conv2d_39_output_array, &conv2d_39_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conversion_40_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 4), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &conversion_40_output_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  nl_41_output, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 4), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &nl_41_output_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  conversion_42_output, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 4), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &conversion_42_output_array, &conversion_42_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  nl_43_output, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 4), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &nl_43_output_array, &nl_43_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_44_output, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 4, 4), AI_STRIDE_INIT(4, 1, 1, 128, 512),
  1, &conv2d_44_output_array, &conv2d_44_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  conversion_45_output, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 4, 4), AI_STRIDE_INIT(4, 4, 4, 512, 2048),
  1, &conversion_45_output_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  nl_46_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 4, 4), AI_STRIDE_INIT(4, 4, 4, 512, 2048),
  1, &nl_46_output_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  conversion_47_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 4, 4), AI_STRIDE_INIT(4, 1, 1, 128, 512),
  1, &conversion_47_output_array, &conversion_47_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  nl_48_output, AI_STATIC,
  56, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 4, 4), AI_STRIDE_INIT(4, 1, 1, 128, 512),
  1, &nl_48_output_array, &nl_48_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  pool_49_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 2, 2), AI_STRIDE_INIT(4, 1, 1, 128, 256),
  1, &pool_49_output_array, &pool_49_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_50_output, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 2, 2), AI_STRIDE_INIT(4, 1, 1, 128, 256),
  1, &conv2d_50_output_array, &conv2d_50_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  conversion_51_output, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 2), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &conversion_51_output_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  nl_52_output, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 2), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &nl_52_output_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  conversion_53_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 2, 2), AI_STRIDE_INIT(4, 1, 1, 128, 256),
  1, &conversion_53_output_array, &conversion_53_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  nl_54_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 2, 2), AI_STRIDE_INIT(4, 1, 1, 128, 256),
  1, &nl_54_output_array, &nl_54_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  transpose_55_output, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 2, 128), AI_STRIDE_INIT(4, 1, 1, 2, 4),
  1, &transpose_55_output_array, &transpose_55_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  transpose_55_output0, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &transpose_55_output_array, &transpose_55_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  dense_57_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &dense_57_output_array, &dense_57_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  conversion_58_output, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &conversion_58_output_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 3, 3, 3, 64), AI_STRIDE_INIT(4, 1, 3, 9, 27),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_weights, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 1, 32), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &conv2d_6_weights_array, &conv2d_6_weights_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_bias, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_6_bias_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_weights, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 1, 32, 96, 288),
  1, &conv2d_11_weights_array, &conv2d_11_weights_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_bias, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_11_bias_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_weights, AI_STATIC,
  73, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 32), AI_STRIDE_INIT(4, 1, 64, 192, 576),
  1, &conv2d_17_weights_array, &conv2d_17_weights_array_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_bias, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_17_bias_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_weights, AI_STATIC,
  75, 0x1,
  AI_SHAPE_INIT(4, 32, 1, 1, 64), AI_STRIDE_INIT(4, 1, 32, 32, 32),
  1, &conv2d_22_weights_array, &conv2d_22_weights_array_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_bias, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_22_bias_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_weights, AI_STATIC,
  77, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 128), AI_STRIDE_INIT(4, 1, 64, 192, 576),
  1, &conv2d_28_weights_array, &conv2d_28_weights_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_bias, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_28_bias_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_weights, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_33_weights_array, &conv2d_33_weights_array_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_bias, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_33_bias_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_weights, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 128, 3, 3, 64), AI_STRIDE_INIT(4, 1, 128, 384, 1152),
  1, &conv2d_39_weights_array, &conv2d_39_weights_array_intq)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_bias, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_39_bias_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_44_weights, AI_STATIC,
  83, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 128), AI_STRIDE_INIT(4, 1, 64, 192, 576),
  1, &conv2d_44_weights_array, &conv2d_44_weights_array_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_44_bias, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_44_bias_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_50_weights, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_50_weights_array, &conv2d_50_weights_array_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_50_bias, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_50_bias_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  dense_57_weights, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 512, 100, 1, 1), AI_STRIDE_INIT(4, 1, 512, 51200, 51200),
  1, &dense_57_weights_array, &dense_57_weights_array_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  dense_57_bias, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &dense_57_bias_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 1, 3820, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3820, 3820),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_scratch0, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &conv2d_6_scratch0_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_scratch0, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 1, 6528, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6528, 6528),
  1, &conv2d_11_scratch0_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_scratch0, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 7552, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7552, 7552),
  1, &conv2d_17_scratch0_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_58_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_57_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_58_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_58_layer, 58,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_58_chain,
  NULL, &conversion_58_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_57_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_55_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_57_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_57_weights, &dense_57_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_57_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_57_layer, 57,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_57_chain,
  NULL, &conversion_58_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_55_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_55_layer, 55,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_55_chain,
  NULL, &dense_57_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_i8 nl_54_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -123, -121, -119, -117, -114, -112, -110, -108, -105, -103, -101, -98, -96, -94, -92, -89, -87, -85, -83, -80, -78, -76, -74, -71, -69, -67, -64, -62, -60, -58, -55, -53, -51, -49, -46, -44, -42, -39, -37, -35, -33, -30, -28, -26, -24, -21, -19, -17, -14, -12, -10, -8, -5, -3, -1, 1, 4, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35, 38, 40, 42, 45, 47, 49, 51, 54, 56, 58, 60, 63, 65, 67, 70, 72, 74, 76, 79, 81, 83, 85, 88, 90, 92, 95, 97, 99, 101, 104, 106, 108, 110, 113, 115, 117, 120, 122, 124, 126 };
AI_ARRAY_OBJ_DECLARE(
    nl_54_nl_params, AI_ARRAY_FORMAT_S8,
    nl_54_nl_params_data, nl_54_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_54_layer, 54,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_54_chain,
  NULL, &transpose_55_layer, AI_STATIC, 
  .nl_params = &nl_54_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_52_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_53_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_53_layer, 53,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_53_chain,
  NULL, &nl_54_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_52_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_51_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_52_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_52_layer, 52,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_52_chain,
  NULL, &conversion_53_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_51_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_51_layer, 51,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_51_chain,
  NULL, &nl_52_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_50_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_50_weights, &conv2d_50_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_50_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_50_layer, 50,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_50_chain,
  NULL, &conversion_51_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_49_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_49_layer, 49,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_49_chain,
  NULL, &conv2d_50_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 nl_48_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -123, -121, -119, -117, -114, -112, -110, -107, -105, -103, -100, -98, -96, -94, -91, -89, -87, -84, -82, -80, -77, -75, -73, -71, -68, -66, -64, -61, -59, -57, -54, -52, -50, -48, -45, -43, -41, -38, -36, -34, -31, -29, -27, -25, -22, -20, -18, -15, -13, -11, -8, -6, -4, -2, 1, 3, 5, 8, 10, 12, 15, 17, 19, 21, 24, 26, 28, 31, 33, 35, 38, 40, 42, 44, 47, 49, 51, 54, 56, 58, 61, 63, 65, 67, 70, 72, 74, 77, 79, 81, 84, 86, 88, 90, 93, 95, 97, 100, 102, 104, 107, 109, 111, 113, 116, 118, 120, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_48_nl_params, AI_ARRAY_FORMAT_S8,
    nl_48_nl_params_data, nl_48_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_47_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_48_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_48_layer, 48,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_48_chain,
  NULL, &pool_49_layer, AI_STATIC, 
  .nl_params = &nl_48_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_47_layer, 47,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_47_chain,
  NULL, &nl_48_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_46_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_46_layer, 46,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_46_chain,
  NULL, &conversion_47_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_44_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_45_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_45_layer, 45,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_45_chain,
  NULL, &nl_46_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_44_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_43_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_44_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_44_weights, &conv2d_44_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_44_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_44_layer, 44,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_44_chain,
  NULL, &conversion_45_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)


AI_STATIC_CONST ai_i8 nl_43_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -122, -120, -117, -114, -111, -109, -106, -103, -100, -97, -95, -92, -89, -86, -84, -81, -78, -75, -73, -70, -67, -64, -61, -59, -56, -53, -50, -48, -45, -42, -39, -36, -34, -31, -28, -25, -23, -20, -17, -14, -11, -9, -6, -3, 0, 2, 5, 8, 11, 13, 16, 19, 22, 25, 27, 30, 33, 36, 38, 41, 44, 47, 50, 52, 55, 58, 61, 63, 66, 69, 72, 74, 77, 80, 83, 86, 88, 91, 94, 97, 99, 102, 105, 108, 111, 113, 116, 119, 122, 124, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_43_nl_params, AI_ARRAY_FORMAT_S8,
    nl_43_nl_params_data, nl_43_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_43_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_43_layer, 43,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_43_chain,
  NULL, &conv2d_44_layer, AI_STATIC, 
  .nl_params = &nl_43_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_42_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_42_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_42_layer, 42,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_42_chain,
  NULL, &nl_43_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_41_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_40_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_41_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_41_layer, 41,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_41_chain,
  NULL, &conversion_42_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_40_layer, 40,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_40_chain,
  NULL, &nl_41_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_39_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_39_weights, &conv2d_39_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_39_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_39_layer, 39,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_39_chain,
  NULL, &conversion_40_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_38_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_38_layer, 38,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_38_chain,
  NULL, &conv2d_39_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 nl_37_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -124, -122, -120, -117, -115, -113, -111, -109, -107, -105, -103, -100, -98, -96, -94, -92, -90, -88, -86, -83, -81, -79, -77, -75, -73, -71, -69, -66, -64, -62, -60, -58, -56, -54, -52, -49, -47, -45, -43, -41, -39, -37, -35, -32, -30, -28, -26, -24, -22, -20, -18, -15, -13, -11, -9, -7, -5, -3, -1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 21, 23, 25, 27, 29, 31, 33, 35, 38, 40, 42, 44, 46, 48, 50, 52, 55, 57, 59, 61, 63, 65, 67, 69, 72, 74, 76, 78, 80, 82, 84, 86, 89, 91, 93, 95, 97, 99, 101, 103, 106, 108, 110, 112, 114, 116, 118, 120, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_37_nl_params, AI_ARRAY_FORMAT_S8,
    nl_37_nl_params_data, nl_37_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_37_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_37_layer, 37,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_37_chain,
  NULL, &pool_38_layer, AI_STATIC, 
  .nl_params = &nl_37_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_36_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_36_layer, 36,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_36_chain,
  NULL, &nl_37_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_35_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_35_layer, 35,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_35_chain,
  NULL, &conversion_36_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_34_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_34_layer, 34,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_34_chain,
  NULL, &nl_35_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_33_weights, &conv2d_33_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_33_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_33_layer, 33,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_33_chain,
  NULL, &conversion_34_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 nl_32_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -124, -121, -119, -117, -115, -112, -110, -108, -106, -103, -101, -99, -97, -94, -92, -90, -88, -85, -83, -81, -79, -76, -74, -72, -70, -68, -65, -63, -61, -59, -56, -54, -52, -50, -47, -45, -43, -41, -38, -36, -34, -32, -29, -27, -25, -23, -20, -18, -16, -14, -12, -9, -7, -5, -3, 0, 2, 4, 6, 9, 11, 13, 15, 18, 20, 22, 24, 27, 29, 31, 33, 36, 38, 40, 42, 44, 47, 49, 51, 53, 56, 58, 60, 62, 65, 67, 69, 71, 74, 76, 78, 80, 83, 85, 87, 89, 92, 94, 96, 98, 100, 103, 105, 107, 109, 112, 114, 116, 118, 121, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_32_nl_params, AI_ARRAY_FORMAT_S8,
    nl_32_nl_params_data, nl_32_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_32_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_32_layer, 32,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_32_chain,
  NULL, &conv2d_33_layer, AI_STATIC, 
  .nl_params = &nl_32_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_30_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_31_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_31_layer, 31,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_31_chain,
  NULL, &nl_32_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_30_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_30_layer, 30,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_30_chain,
  NULL, &conversion_31_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_29_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_29_layer, 29,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_29_chain,
  NULL, &nl_30_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_28_weights, &conv2d_28_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_28_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_28_layer, 28,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_28_chain,
  NULL, &conversion_29_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_27_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_27_layer, 27,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_27_chain,
  NULL, &conv2d_28_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 nl_26_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -123, -121, -119, -117, -114, -112, -110, -107, -105, -103, -100, -98, -96, -94, -91, -89, -87, -84, -82, -80, -77, -75, -73, -71, -68, -66, -64, -61, -59, -57, -54, -52, -50, -48, -45, -43, -41, -38, -36, -34, -32, -29, -27, -25, -22, -20, -18, -15, -13, -11, -9, -6, -4, -2, 1, 3, 5, 8, 10, 12, 14, 17, 19, 21, 24, 26, 28, 30, 33, 35, 37, 40, 42, 44, 47, 49, 51, 53, 56, 58, 60, 63, 65, 67, 70, 72, 74, 76, 79, 81, 83, 86, 88, 90, 93, 95, 97, 99, 102, 104, 106, 109, 111, 113, 115, 118, 120, 122, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_26_nl_params, AI_ARRAY_FORMAT_S8,
    nl_26_nl_params_data, nl_26_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_26_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_26_layer, 26,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_26_chain,
  NULL, &pool_27_layer, AI_STATIC, 
  .nl_params = &nl_26_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_25_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_25_chain,
  NULL, &nl_26_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_24_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_24_chain,
  NULL, &conversion_25_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_23_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_23_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_23_chain,
  NULL, &nl_24_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_22_weights, &conv2d_22_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_22_layer, 22,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_22_chain,
  NULL, &conversion_23_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 nl_21_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -124, -122, -120, -118, -116, -114, -112, -110, -109, -107, -105, -103, -101, -99, -97, -95, -93, -91, -89, -87, -85, -83, -81, -79, -77, -75, -73, -72, -70, -68, -66, -64, -62, -60, -58, -56, -54, -52, -50, -48, -46, -44, -42, -40, -38, -36, -34, -33, -31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 117, 119, 121, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_21_nl_params, AI_ARRAY_FORMAT_S8,
    nl_21_nl_params_data, nl_21_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_21_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_21_layer, 21,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_21_chain,
  NULL, &conv2d_22_layer, AI_STATIC, 
  .nl_params = &nl_21_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_20_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_20_layer, 20,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_20_chain,
  NULL, &nl_21_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_19_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_19_layer, 19,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_19_chain,
  NULL, &conversion_20_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_18_layer, 18,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_18_chain,
  NULL, &nl_19_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_17_weights, &conv2d_17_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_17_layer, 17,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_17_chain,
  NULL, &conversion_18_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_16_layer, 16,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_16_chain,
  NULL, &conv2d_17_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 nl_15_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -123, -121, -119, -116, -114, -112, -110, -107, -105, -103, -100, -98, -96, -93, -91, -89, -87, -84, -82, -80, -77, -75, -73, -70, -68, -66, -63, -61, -59, -57, -54, -52, -50, -47, -45, -43, -40, -38, -36, -34, -31, -29, -27, -24, -22, -20, -17, -15, -13, -11, -8, -6, -4, -1, 1, 3, 6, 8, 10, 13, 15, 17, 19, 22, 24, 26, 29, 31, 33, 36, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66, 68, 70, 72, 75, 77, 79, 82, 84, 86, 89, 91, 93, 95, 98, 100, 102, 105, 107, 109, 112, 114, 116, 118, 121, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_15_nl_params, AI_ARRAY_FORMAT_S8,
    nl_15_nl_params_data, nl_15_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_15_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_15_chain,
  NULL, &pool_16_layer, AI_STATIC, 
  .nl_params = &nl_15_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_14_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_14_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_14_chain,
  NULL, &nl_15_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_13_layer, 13,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_13_chain,
  NULL, &conversion_14_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_12_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_12_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_12_chain,
  NULL, &nl_13_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_11_weights, &conv2d_11_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_11_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_11_chain,
  NULL, &conversion_12_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)


AI_STATIC_CONST ai_i8 nl_10_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -124, -123, -121, -119, -117, -116, -114, -112, -110, -109, -107, -105, -103, -102, -100, -98, -96, -95, -93, -91, -89, -87, -86, -84, -82, -80, -79, -77, -75, -73, -72, -70, -68, -66, -65, -63, -61, -59, -58, -56, -54, -52, -50, -49, -47, -45, -43, -42, -40, -38, -36, -35, -33, -31, -29, -28, -26, -24, -22, -21, -19, -17, -15, -13, -12, -10, -8, -6, -5, -3, -1, 1, 2, 4, 6, 8, 9, 11, 13, 15, 16, 18, 20, 22, 24, 25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 43, 45, 46, 48, 50, 52, 53, 55, 57, 59, 61, 62, 64, 66, 68, 69, 71, 73, 75, 76, 78, 80, 82, 83, 85, 87, 89, 90, 92, 94, 96, 98, 99, 101, 103, 105, 106, 108, 110, 112, 113, 115, 117, 119, 120, 122, 124, 126, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_10_nl_params, AI_ARRAY_FORMAT_S8,
    nl_10_nl_params_data, nl_10_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_10_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_10_chain,
  NULL, &conv2d_11_layer, AI_STATIC, 
  .nl_params = &nl_10_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_9_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_9_chain,
  NULL, &nl_10_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_8_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_8_chain,
  NULL, &conversion_9_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_7_layer, 7,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_7_chain,
  NULL, &nl_8_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_6_weights, &conv2d_6_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_6_layer, 6,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_6_chain,
  NULL, &conversion_7_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 nl_5_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -124, -122, -120, -118, -116, -114, -112, -110, -107, -105, -103, -101, -99, -97, -95, -93, -91, -89, -87, -85, -83, -81, -79, -77, -75, -73, -70, -68, -66, -64, -62, -60, -58, -56, -54, -52, -50, -48, -46, -44, -42, -40, -38, -36, -34, -31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 121, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_5_nl_params, AI_ARRAY_FORMAT_S8,
    nl_5_nl_params_data, nl_5_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_5_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_5_chain,
  NULL, &conv2d_6_layer, AI_STATIC, 
  .nl_params = &nl_5_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_4_layer, 4,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_4_chain,
  NULL, &nl_5_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_3_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &nl_3_chain,
  NULL, &conversion_4_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_2_layer, 2,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_2_chain,
  NULL, &nl_3_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_1_chain,
  NULL, &conversion_2_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_x_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_0_layer, 0,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_0_chain,
  NULL, &conv2d_1_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 351568, 1, 1),
    351568, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 327680, 1, 1),
    327680, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_x_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &conversion_58_output),
  &conversion_0_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 351568, 1, 1),
      351568, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 327680, 1, 1),
      327680, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_x_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &conversion_58_output),
  &conversion_0_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_x_10_output_array.data = AI_PTR(g_network_activations_map[0] + 66684);
    serving_default_x_10_output_array.data_start = AI_PTR(g_network_activations_map[0] + 66684);
    
    conversion_0_output_array.data = AI_PTR(g_network_activations_map[0] + 66684);
    conversion_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 66684);
    
    conv2d_1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 69760);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 69760);
    
    conv2d_1_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conversion_2_output_array.data = AI_PTR(g_network_activations_map[0] + 65536);
    conversion_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 65536);
    
    nl_3_output_array.data = AI_PTR(g_network_activations_map[0] + 65536);
    nl_3_output_array.data_start = AI_PTR(g_network_activations_map[0] + 65536);
    
    conversion_4_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_4_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_5_output_array.data = AI_PTR(g_network_activations_map[0] + 65536);
    nl_5_output_array.data_start = AI_PTR(g_network_activations_map[0] + 65536);
    
    conv2d_6_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_6_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_6_output_array.data = AI_PTR(g_network_activations_map[0] + 256);
    conv2d_6_output_array.data_start = AI_PTR(g_network_activations_map[0] + 256);
    
    conversion_7_output_array.data = AI_PTR(g_network_activations_map[0] + 33024);
    conversion_7_output_array.data_start = AI_PTR(g_network_activations_map[0] + 33024);
    
    nl_8_output_array.data = AI_PTR(g_network_activations_map[0] + 164096);
    nl_8_output_array.data_start = AI_PTR(g_network_activations_map[0] + 164096);
    
    conversion_9_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_9_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_10_output_array.data = AI_PTR(g_network_activations_map[0] + 32768);
    nl_10_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32768);
    
    conv2d_11_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_11_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_11_output_array.data = AI_PTR(g_network_activations_map[0] + 262144);
    conv2d_11_output_array.data_start = AI_PTR(g_network_activations_map[0] + 262144);
    
    conversion_12_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_12_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_13_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_13_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conversion_14_output_array.data = AI_PTR(g_network_activations_map[0] + 262144);
    conversion_14_output_array.data_start = AI_PTR(g_network_activations_map[0] + 262144);
    
    nl_15_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_15_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    pool_16_output_array.data = AI_PTR(g_network_activations_map[0] + 65536);
    pool_16_output_array.data_start = AI_PTR(g_network_activations_map[0] + 65536);
    
    conv2d_17_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_17_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_17_output_array.data = AI_PTR(g_network_activations_map[0] + 7552);
    conv2d_17_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7552);
    
    conversion_18_output_array.data = AI_PTR(g_network_activations_map[0] + 15744);
    conversion_18_output_array.data_start = AI_PTR(g_network_activations_map[0] + 15744);
    
    nl_19_output_array.data = AI_PTR(g_network_activations_map[0] + 48512);
    nl_19_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48512);
    
    conversion_20_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_20_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_21_output_array.data = AI_PTR(g_network_activations_map[0] + 8192);
    nl_21_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8192);
    
    conv2d_22_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_22_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_22_output_array.data = AI_PTR(g_network_activations_map[0] + 16384);
    conv2d_22_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16384);
    
    conversion_23_output_array.data = AI_PTR(g_network_activations_map[0] + 32768);
    conversion_23_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32768);
    
    nl_24_output_array.data = AI_PTR(g_network_activations_map[0] + 98304);
    nl_24_output_array.data_start = AI_PTR(g_network_activations_map[0] + 98304);
    
    conversion_25_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_25_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_26_output_array.data = AI_PTR(g_network_activations_map[0] + 16384);
    nl_26_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16384);
    
    pool_27_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    pool_27_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_28_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 4096);
    conv2d_28_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 4096);
    
    conv2d_28_output_array.data = AI_PTR(g_network_activations_map[0] + 12032);
    conv2d_28_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12032);
    
    conversion_29_output_array.data = AI_PTR(g_network_activations_map[0] + 20224);
    conversion_29_output_array.data_start = AI_PTR(g_network_activations_map[0] + 20224);
    
    nl_30_output_array.data = AI_PTR(g_network_activations_map[0] + 52992);
    nl_30_output_array.data_start = AI_PTR(g_network_activations_map[0] + 52992);
    
    conversion_31_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_31_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_32_output_array.data = AI_PTR(g_network_activations_map[0] + 8192);
    nl_32_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8192);
    
    conv2d_33_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_33_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_33_output_array.data = AI_PTR(g_network_activations_map[0] + 16384);
    conv2d_33_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16384);
    
    conversion_34_output_array.data = AI_PTR(g_network_activations_map[0] + 24576);
    conversion_34_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24576);
    
    nl_35_output_array.data = AI_PTR(g_network_activations_map[0] + 57344);
    nl_35_output_array.data_start = AI_PTR(g_network_activations_map[0] + 57344);
    
    conversion_36_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_36_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_37_output_array.data = AI_PTR(g_network_activations_map[0] + 8192);
    nl_37_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8192);
    
    pool_38_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    pool_38_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_39_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2048);
    conv2d_39_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2048);
    
    conv2d_39_output_array.data = AI_PTR(g_network_activations_map[0] + 12032);
    conv2d_39_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12032);
    
    conversion_40_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_40_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_41_output_array.data = AI_PTR(g_network_activations_map[0] + 4096);
    nl_41_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4096);
    
    conversion_42_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_42_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_43_output_array.data = AI_PTR(g_network_activations_map[0] + 1024);
    nl_43_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024);
    
    conv2d_44_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2048);
    conv2d_44_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2048);
    
    conv2d_44_output_array.data = AI_PTR(g_network_activations_map[0] + 9984);
    conv2d_44_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9984);
    
    conversion_45_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_45_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_46_output_array.data = AI_PTR(g_network_activations_map[0] + 8192);
    nl_46_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8192);
    
    conversion_47_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_47_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_48_output_array.data = AI_PTR(g_network_activations_map[0] + 2048);
    nl_48_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2048);
    
    pool_49_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    pool_49_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    conv2d_50_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 512);
    conv2d_50_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    
    conv2d_50_output_array.data = AI_PTR(g_network_activations_map[0] + 1024);
    conv2d_50_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024);
    
    conversion_51_output_array.data = AI_PTR(g_network_activations_map[0] + 1536);
    conversion_51_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1536);
    
    nl_52_output_array.data = AI_PTR(g_network_activations_map[0] + 3584);
    nl_52_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3584);
    
    conversion_53_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_53_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    nl_54_output_array.data = AI_PTR(g_network_activations_map[0] + 512);
    nl_54_output_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    
    transpose_55_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    transpose_55_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_57_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 512);
    dense_57_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    
    dense_57_output_array.data = AI_PTR(g_network_activations_map[0] + 1536);
    dense_57_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1536);
    
    conversion_58_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conversion_58_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
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
    conv2d_1_bias_array.data = AI_PTR(g_network_weights_map[0] + 1728);
    conv2d_1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1728);
    
    conv2d_6_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_weights_array.data = AI_PTR(g_network_weights_map[0] + 1984);
    conv2d_6_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1984);
    
    conv2d_6_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_bias_array.data = AI_PTR(g_network_weights_map[0] + 4032);
    conv2d_6_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4032);
    
    conv2d_11_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_weights_array.data = AI_PTR(g_network_weights_map[0] + 4160);
    conv2d_11_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 4160);
    
    conv2d_11_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_bias_array.data = AI_PTR(g_network_weights_map[0] + 22592);
    conv2d_11_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 22592);
    
    conv2d_17_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_weights_array.data = AI_PTR(g_network_weights_map[0] + 22848);
    conv2d_17_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 22848);
    
    conv2d_17_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_bias_array.data = AI_PTR(g_network_weights_map[0] + 41280);
    conv2d_17_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 41280);
    
    conv2d_22_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_weights_array.data = AI_PTR(g_network_weights_map[0] + 41408);
    conv2d_22_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 41408);
    
    conv2d_22_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_bias_array.data = AI_PTR(g_network_weights_map[0] + 43456);
    conv2d_22_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 43456);
    
    conv2d_28_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_28_weights_array.data = AI_PTR(g_network_weights_map[0] + 43712);
    conv2d_28_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 43712);
    
    conv2d_28_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_28_bias_array.data = AI_PTR(g_network_weights_map[0] + 117440);
    conv2d_28_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 117440);
    
    conv2d_33_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_33_weights_array.data = AI_PTR(g_network_weights_map[0] + 117952);
    conv2d_33_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 117952);
    
    conv2d_33_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_33_bias_array.data = AI_PTR(g_network_weights_map[0] + 134336);
    conv2d_33_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 134336);
    
    conv2d_39_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_39_weights_array.data = AI_PTR(g_network_weights_map[0] + 134848);
    conv2d_39_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 134848);
    
    conv2d_39_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_39_bias_array.data = AI_PTR(g_network_weights_map[0] + 208576);
    conv2d_39_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 208576);
    
    conv2d_44_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_44_weights_array.data = AI_PTR(g_network_weights_map[0] + 208832);
    conv2d_44_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 208832);
    
    conv2d_44_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_44_bias_array.data = AI_PTR(g_network_weights_map[0] + 282560);
    conv2d_44_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 282560);
    
    conv2d_50_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_50_weights_array.data = AI_PTR(g_network_weights_map[0] + 283072);
    conv2d_50_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 283072);
    
    conv2d_50_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_50_bias_array.data = AI_PTR(g_network_weights_map[0] + 299456);
    conv2d_50_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 299456);
    
    dense_57_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_57_weights_array.data = AI_PTR(g_network_weights_map[0] + 299968);
    dense_57_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 299968);
    
    dense_57_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_57_bias_array.data = AI_PTR(g_network_weights_map[0] + 351168);
    dense_57_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 351168);
    
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
      
      .n_macc            = 37785196,
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
      
      .n_macc            = 37785196,
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

