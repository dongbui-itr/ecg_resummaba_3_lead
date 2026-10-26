/**
  ******************************************************************************
  * @file    ecga.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-25T22:15:20+0700
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "ecga.h"
#include "ecga_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_ecga
 
#undef AI_ECGA_MODEL_SIGNATURE
#define AI_ECGA_MODEL_SIGNATURE     "0xd59a69422bb3f893c887b6ff86f3a54b"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-10-25T22:15:20+0700"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_ECGA_N_BATCHES
#define AI_ECGA_N_BATCHES         (1)

static ai_ptr g_ecga_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_ecga_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  xs_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 7500, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  xs_0_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30000, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  nl_3_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30000, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30000, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  nl_4_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30000, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  pool_25_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10000, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  pool_24_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10000, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  pool_13_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 600, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 600, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  nl_14_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 600, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  transpose_16_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  nl_17_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  transpose_18_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  tile_20_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_22_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30000, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  reduce_23_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10000, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  concat_26_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30000, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  pool_28_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15000, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  nl_29_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_30_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  nl_30_nl_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 7500, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  pool_32_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  nl_33_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_34_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  nl_34_nl_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2000, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_54_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  nl_54_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_59_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_60_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  nl_62_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_63_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_65_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  nl_57_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_58_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_67_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_69_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_70_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  nl_70_nl_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2500, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  pool_36_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 400, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  nl_37_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_38_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  nl_38_nl_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 500, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  pool_40_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 120, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  nl_41_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 120, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_42_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 120, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  nl_42_nl_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 120, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  tile_45_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 600, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  concat_47_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1100, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_49_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  nl_49_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_50_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  nl_50_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  tile_52_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  concat_72_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7000, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_74_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  nl_74_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_75_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  nl_75_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  tile_77_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10000, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  concat_79_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 17500, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_81_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  nl_81_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_82_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  nl_82_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_83_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1250, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  nl_86_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1250, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6000, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1200, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 300, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  pool_11_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 60, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 60, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  nl_12_nl_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 60, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 504, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_30_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 252, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_30_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 240, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_34_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 320, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_34_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_54_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 80, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_54_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_59_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_59_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_60_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_60_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_65_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2550, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_69_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_69_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 400, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_38_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_38_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 360, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_42_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 432, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_42_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_49_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1100, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_49_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_50_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_50_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_74_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1120, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_74_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_75_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 320, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_75_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_81_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 588, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_81_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_82_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 252, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_82_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_83_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_83_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 84, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_30_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 42, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_34_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_54_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_59_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_69_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_38_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_42_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_49_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 110, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_50_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_74_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 140, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_75_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_81_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 98, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_82_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 42, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_83_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 28, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10000, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 28, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 28, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 400, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  concat_26_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 2500), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &concat_26_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  concat_26_output0, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 2500, 1), AI_STRIDE_INIT(4, 4, 4, 48, 120000),
  1, &concat_26_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  concat_47_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 1, 50), AI_STRIDE_INIT(4, 4, 4, 88, 88),
  1, &concat_47_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  concat_47_output0, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 50, 1), AI_STRIDE_INIT(4, 4, 4, 88, 4400),
  1, &concat_47_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  concat_72_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 28, 1, 250), AI_STRIDE_INIT(4, 4, 4, 112, 112),
  1, &concat_72_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  concat_72_output0, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 28, 250, 1), AI_STRIDE_INIT(4, 4, 4, 112, 28000),
  1, &concat_72_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  concat_79_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 14, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 56, 56),
  1, &concat_79_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  concat_79_output0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 14, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 56, 70000),
  1, &concat_79_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_29_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &conv2d_29_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_scratch0, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 7, 1), AI_STRIDE_INIT(4, 4, 4, 48, 336),
  1, &conv2d_29_scratch0_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 12, 7, 1, 6), AI_STRIDE_INIT(4, 4, 48, 288, 2016),
  1, &conv2d_29_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_30_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_30_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_30_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &conv2d_30_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_30_scratch0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 7, 1), AI_STRIDE_INIT(4, 4, 4, 24, 168),
  1, &conv2d_30_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_30_weights, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 6, 7, 1, 6), AI_STRIDE_INIT(4, 4, 24, 144, 1008),
  1, &conv2d_30_weights_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_33_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 8000),
  1, &conv2d_33_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 5, 1), AI_STRIDE_INIT(4, 4, 4, 24, 120),
  1, &conv2d_33_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_weights, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 6, 5, 1, 8), AI_STRIDE_INIT(4, 4, 24, 192, 960),
  1, &conv2d_33_weights_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_34_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_34_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_34_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 8000),
  1, &conv2d_34_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_34_scratch0, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 5, 1), AI_STRIDE_INIT(4, 4, 4, 32, 160),
  1, &conv2d_34_scratch0_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_34_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 8, 5, 1, 8), AI_STRIDE_INIT(4, 4, 32, 256, 1280),
  1, &conv2d_34_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_37_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &conv2d_37_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_scratch0, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 5, 1), AI_STRIDE_INIT(4, 4, 4, 32, 160),
  1, &conv2d_37_scratch0_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_weights, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 8, 5, 1, 10), AI_STRIDE_INIT(4, 4, 32, 320, 1600),
  1, &conv2d_37_weights_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_38_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_38_bias_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_38_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &conv2d_38_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_38_scratch0, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 5, 1), AI_STRIDE_INIT(4, 4, 4, 40, 200),
  1, &conv2d_38_scratch0_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_38_weights, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 10, 5, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 2000),
  1, &conv2d_38_weights_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_bias, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_3_bias_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 2500, 3), AI_STRIDE_INIT(4, 4, 4, 16, 40000),
  1, &conv2d_3_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_scratch0, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 9, 1), AI_STRIDE_INIT(4, 4, 4, 4, 36),
  1, &conv2d_3_scratch0_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_weights, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 9, 1, 4), AI_STRIDE_INIT(4, 4, 4, 16, 144),
  1, &conv2d_3_weights_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_bias, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 1), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &conv2d_41_bias_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 10, 1), AI_STRIDE_INIT(4, 4, 4, 48, 480),
  1, &conv2d_41_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_scratch0, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 3, 1), AI_STRIDE_INIT(4, 4, 4, 40, 120),
  1, &conv2d_41_scratch0_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_weights, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 10, 3, 1, 12), AI_STRIDE_INIT(4, 4, 40, 480, 1440),
  1, &conv2d_41_weights_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_42_bias, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 1), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &conv2d_42_bias_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_42_output, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 10, 1), AI_STRIDE_INIT(4, 4, 4, 48, 480),
  1, &conv2d_42_output_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_42_scratch0, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 3, 1), AI_STRIDE_INIT(4, 4, 4, 48, 144),
  1, &conv2d_42_scratch0_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_42_weights, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 12, 3, 1, 12), AI_STRIDE_INIT(4, 4, 48, 576, 1728),
  1, &conv2d_42_weights_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_49_bias, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_49_bias_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_49_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &conv2d_49_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_49_scratch0, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 5, 1), AI_STRIDE_INIT(4, 4, 4, 88, 440),
  1, &conv2d_49_scratch0_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_49_weights, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 22, 5, 1, 10), AI_STRIDE_INIT(4, 4, 88, 880, 4400),
  1, &conv2d_49_weights_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 2500, 3), AI_STRIDE_INIT(4, 4, 4, 16, 40000),
  1, &conv2d_4_output_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 9, 1), AI_STRIDE_INIT(4, 4, 4, 16, 144),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 4, 9, 1, 4), AI_STRIDE_INIT(4, 4, 16, 64, 576),
  1, &conv2d_4_weights_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_50_bias, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_50_bias_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_50_output, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &conv2d_50_output_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_50_scratch0, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 5, 1), AI_STRIDE_INIT(4, 4, 4, 40, 200),
  1, &conv2d_50_scratch0_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_50_weights, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 10, 5, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 2000),
  1, &conv2d_50_weights_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_54_bias, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_54_bias_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_54_output, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_54_output_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_54_scratch0, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_54_scratch0_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_54_weights, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 10), AI_STRIDE_INIT(4, 4, 32, 320, 320),
  1, &conv2d_54_weights_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_output, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_55_output_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_output0, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_55_output_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_scratch0, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_55_scratch0_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_weights, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 10, 1, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 400),
  1, &conv2d_55_weights_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_59_bias, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_59_bias_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_59_output, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_59_output_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_59_scratch0, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_59_scratch0_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_59_weights, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 10, 1, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 400),
  1, &conv2d_59_weights_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_bias, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_5_bias_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 500, 3), AI_STRIDE_INIT(4, 4, 4, 16, 8000),
  1, &conv2d_5_output_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_scratch0, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 7, 1), AI_STRIDE_INIT(4, 4, 4, 16, 112),
  1, &conv2d_5_scratch0_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_scratch1, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 2500, 1), AI_STRIDE_INIT(4, 4, 4, 16, 40000),
  1, &conv2d_5_scratch1_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_weights, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 4, 7, 1, 4), AI_STRIDE_INIT(4, 4, 16, 64, 448),
  1, &conv2d_5_weights_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_60_bias, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_60_bias_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_60_output, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_60_output_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_60_output0, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_60_output_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_60_weights, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 10), AI_STRIDE_INIT(4, 1, 10, 10, 10),
  1, &conv2d_60_weights_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_65_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_65_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_65_output0, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_65_output_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_65_weights, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 1, 255, 1, 10), AI_STRIDE_INIT(4, 1, 10, 10, 10),
  1, &conv2d_65_weights_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_69_bias, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_69_bias_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_69_output, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_69_output_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_69_scratch0, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_69_scratch0_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_69_weights, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 10, 1, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 400),
  1, &conv2d_69_weights_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_74_bias, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_74_bias_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_74_output, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 8000),
  1, &conv2d_74_output_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_74_scratch0, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 28, 5, 1), AI_STRIDE_INIT(4, 4, 4, 112, 560),
  1, &conv2d_74_scratch0_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_74_weights, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 28, 5, 1, 8), AI_STRIDE_INIT(4, 4, 112, 896, 4480),
  1, &conv2d_74_weights_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_75_bias, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_75_bias_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_75_output, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 8000),
  1, &conv2d_75_output_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_75_scratch0, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 5, 1), AI_STRIDE_INIT(4, 4, 4, 32, 160),
  1, &conv2d_75_scratch0_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_75_weights, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 8, 5, 1, 8), AI_STRIDE_INIT(4, 4, 32, 256, 1280),
  1, &conv2d_75_weights_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 100, 3), AI_STRIDE_INIT(4, 4, 4, 16, 1600),
  1, &conv2d_7_output_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  94, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 7, 1), AI_STRIDE_INIT(4, 4, 4, 16, 112),
  1, &conv2d_7_scratch0_array, NULL)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch1, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 500, 1), AI_STRIDE_INIT(4, 4, 4, 16, 8000),
  1, &conv2d_7_scratch1_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  96, 0x0,
  AI_SHAPE_INIT(4, 4, 7, 1, 4), AI_STRIDE_INIT(4, 4, 16, 64, 448),
  1, &conv2d_7_weights_array, NULL)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_81_bias, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_81_bias_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_81_output, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &conv2d_81_output_array, NULL)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_81_scratch0, AI_STATIC,
  99, 0x0,
  AI_SHAPE_INIT(4, 1, 14, 7, 1), AI_STRIDE_INIT(4, 4, 4, 56, 392),
  1, &conv2d_81_scratch0_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_81_weights, AI_STATIC,
  100, 0x0,
  AI_SHAPE_INIT(4, 14, 7, 1, 6), AI_STRIDE_INIT(4, 4, 56, 336, 2352),
  1, &conv2d_81_weights_array, NULL)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_82_bias, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_82_bias_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_82_output, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &conv2d_82_output_array, NULL)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_82_scratch0, AI_STATIC,
  103, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 7, 1), AI_STRIDE_INIT(4, 4, 4, 24, 168),
  1, &conv2d_82_scratch0_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_82_weights, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 6, 7, 1, 6), AI_STRIDE_INIT(4, 4, 24, 144, 1008),
  1, &conv2d_82_weights_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_83_bias, AI_STATIC,
  105, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_83_bias_array, NULL)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_83_output, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 4, 5000),
  1, &conv2d_83_output_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_83_output0, AI_STATIC,
  107, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_83_output_array, NULL)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_83_scratch0, AI_STATIC,
  108, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_83_scratch0_array, NULL)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_83_weights, AI_STATIC,
  109, 0x0,
  AI_SHAPE_INIT(4, 6, 1, 1, 1), AI_STRIDE_INIT(4, 4, 24, 24, 24),
  1, &conv2d_83_weights_array, NULL)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_bias, AI_STATIC,
  110, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_9_bias_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_output, AI_STATIC,
  111, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 25, 3), AI_STRIDE_INIT(4, 4, 4, 16, 400),
  1, &conv2d_9_output_array, NULL)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_scratch0, AI_STATIC,
  112, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 7, 1), AI_STRIDE_INIT(4, 4, 4, 16, 112),
  1, &conv2d_9_scratch0_array, NULL)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_scratch1, AI_STATIC,
  113, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 100, 1), AI_STRIDE_INIT(4, 4, 4, 16, 1600),
  1, &conv2d_9_scratch1_array, NULL)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_weights, AI_STATIC,
  114, 0x0,
  AI_SHAPE_INIT(4, 4, 7, 1, 4), AI_STRIDE_INIT(4, 4, 16, 64, 448),
  1, &conv2d_9_weights_array, NULL)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_22_output, AI_STATIC,
  115, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 2500, 3), AI_STRIDE_INIT(4, 4, 4, 16, 40000),
  1, &eltwise_22_output_array, NULL)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_58_output, AI_STATIC,
  116, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_58_output_array, NULL)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_63_output, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_63_output_array, NULL)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_63_output0, AI_STATIC,
  118, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &eltwise_63_output_array, NULL)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_67_output, AI_STATIC,
  119, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_67_output_array, NULL)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_67_output0, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &eltwise_67_output_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_70_output, AI_STATIC,
  121, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &eltwise_70_output_array, NULL)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_bias, AI_STATIC,
  122, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_12_bias_array, NULL)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_output, AI_STATIC,
  123, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 5, 3), AI_STRIDE_INIT(4, 4, 4, 16, 80),
  1, &gemm_12_output_array, NULL)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_weights, AI_STATIC,
  124, 0x0,
  AI_SHAPE_INIT(4, 4, 4, 1, 1), AI_STRIDE_INIT(4, 4, 16, 64, 64),
  1, &gemm_12_weights_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_bias, AI_STATIC,
  125, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_14_bias_array, NULL)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_output, AI_STATIC,
  126, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 50, 3), AI_STRIDE_INIT(4, 4, 4, 16, 800),
  1, &gemm_14_output_array, NULL)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_weights, AI_STATIC,
  127, 0x0,
  AI_SHAPE_INIT(4, 4, 4, 1, 1), AI_STRIDE_INIT(4, 4, 16, 64, 64),
  1, &gemm_14_weights_array, NULL)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_bias, AI_STATIC,
  128, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_15_bias_array, NULL)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_output, AI_STATIC,
  129, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 50, 3), AI_STRIDE_INIT(4, 4, 4, 4, 200),
  1, &gemm_15_output_array, NULL)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_weights, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 4, 1, 1, 1), AI_STRIDE_INIT(4, 4, 16, 16, 16),
  1, &gemm_15_weights_array, NULL)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  nl_12_nl_output, AI_STATIC,
  131, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 5, 3), AI_STRIDE_INIT(4, 4, 4, 16, 80),
  1, &nl_12_nl_output_array, NULL)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  nl_14_nl_output, AI_STATIC,
  132, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 50, 3), AI_STRIDE_INIT(4, 4, 4, 16, 800),
  1, &nl_14_nl_output_array, NULL)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  nl_17_output, AI_STATIC,
  133, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 50, 1), AI_STRIDE_INIT(4, 4, 4, 12, 600),
  1, &nl_17_output_array, NULL)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  nl_29_nl_output, AI_STATIC,
  134, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &nl_29_nl_output_array, NULL)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  nl_30_nl_output, AI_STATIC,
  135, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &nl_30_nl_output_array, NULL)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  nl_30_nl_output0, AI_STATIC,
  136, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &nl_30_nl_output_array, NULL)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  nl_33_nl_output, AI_STATIC,
  137, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 8000),
  1, &nl_33_nl_output_array, NULL)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  nl_34_nl_output, AI_STATIC,
  138, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 8000),
  1, &nl_34_nl_output_array, NULL)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  nl_34_nl_output0, AI_STATIC,
  139, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 250), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &nl_34_nl_output_array, NULL)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  nl_37_nl_output, AI_STATIC,
  140, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &nl_37_nl_output_array, NULL)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  nl_38_nl_output, AI_STATIC,
  141, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &nl_38_nl_output_array, NULL)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  nl_38_nl_output0, AI_STATIC,
  142, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 50), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_38_nl_output_array, NULL)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  nl_3_nl_output, AI_STATIC,
  143, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 2500, 3), AI_STRIDE_INIT(4, 4, 4, 16, 40000),
  1, &nl_3_nl_output_array, NULL)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  nl_41_nl_output, AI_STATIC,
  144, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 10, 1), AI_STRIDE_INIT(4, 4, 4, 48, 480),
  1, &nl_41_nl_output_array, NULL)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  nl_42_nl_output, AI_STATIC,
  145, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 10, 1), AI_STRIDE_INIT(4, 4, 4, 48, 480),
  1, &nl_42_nl_output_array, NULL)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  nl_42_nl_output0, AI_STATIC,
  146, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 10), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &nl_42_nl_output_array, NULL)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  nl_49_nl_output, AI_STATIC,
  147, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &nl_49_nl_output_array, NULL)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  nl_4_nl_output, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 2500, 3), AI_STRIDE_INIT(4, 4, 4, 16, 40000),
  1, &nl_4_nl_output_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  nl_50_nl_output, AI_STATIC,
  149, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &nl_50_nl_output_array, NULL)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  nl_50_nl_output0, AI_STATIC,
  150, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 50), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_50_nl_output_array, NULL)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  nl_54_nl_output, AI_STATIC,
  151, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &nl_54_nl_output_array, NULL)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  nl_57_output, AI_STATIC,
  152, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_57_output_array, NULL)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  nl_62_output, AI_STATIC,
  153, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_62_output_array, NULL)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  nl_70_nl_output, AI_STATIC,
  154, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &nl_70_nl_output_array, NULL)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  nl_70_nl_output0, AI_STATIC,
  155, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_70_nl_output_array, NULL)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  nl_74_nl_output, AI_STATIC,
  156, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 8000),
  1, &nl_74_nl_output_array, NULL)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  nl_75_nl_output, AI_STATIC,
  157, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 8000),
  1, &nl_75_nl_output_array, NULL)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  nl_75_nl_output0, AI_STATIC,
  158, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 250), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &nl_75_nl_output_array, NULL)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  nl_81_nl_output, AI_STATIC,
  159, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &nl_81_nl_output_array, NULL)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  nl_82_nl_output, AI_STATIC,
  160, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &nl_82_nl_output_array, NULL)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  nl_86_output, AI_STATIC,
  161, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_86_output_array, NULL)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  pool_11_output, AI_STATIC,
  162, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 5, 3), AI_STRIDE_INIT(4, 4, 4, 16, 80),
  1, &pool_11_output_array, NULL)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  pool_13_output, AI_STATIC,
  163, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 50, 3), AI_STRIDE_INIT(4, 4, 4, 16, 800),
  1, &pool_13_output_array, NULL)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  pool_24_output, AI_STATIC,
  164, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 2500, 1), AI_STRIDE_INIT(4, 4, 4, 16, 40000),
  1, &pool_24_output_array, NULL)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  pool_24_output0, AI_STATIC,
  165, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 2500), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &pool_24_output_array, NULL)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  pool_25_output, AI_STATIC,
  166, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 2500, 1), AI_STRIDE_INIT(4, 4, 4, 16, 40000),
  1, &pool_25_output_array, NULL)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  pool_25_output0, AI_STATIC,
  167, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 2500), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &pool_25_output_array, NULL)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  pool_28_output, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 48, 60000),
  1, &pool_28_output_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  pool_32_output, AI_STATIC,
  169, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &pool_32_output_array, NULL)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  pool_36_output, AI_STATIC,
  170, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 50, 1), AI_STRIDE_INIT(4, 4, 4, 32, 1600),
  1, &pool_36_output_array, NULL)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  pool_40_output, AI_STATIC,
  171, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 10, 1), AI_STRIDE_INIT(4, 4, 4, 40, 400),
  1, &pool_40_output_array, NULL)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  reduce_23_output, AI_STATIC,
  172, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 2500, 1), AI_STRIDE_INIT(4, 4, 4, 16, 40000),
  1, &reduce_23_output_array, NULL)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  reduce_23_output0, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 2500), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &reduce_23_output_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  tile_20_output, AI_STATIC,
  174, 0x0,
  AI_SHAPE_INIT(5, 1, 1, 50, 3, 50), AI_STRIDE_INIT(5, 4, 4, 200, 10000, 4),
  1, &tile_20_output_array, NULL)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  tile_20_output0, AI_STATIC,
  175, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 2500, 3), AI_STRIDE_INIT(4, 4, 4, 4, 10000),
  1, &tile_20_output_array, NULL)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  tile_45_output, AI_STATIC,
  176, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 5, 10), AI_STRIDE_INIT(4, 4, 4, 48, 240),
  1, &tile_45_output_array, NULL)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  tile_45_output0, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 50), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &tile_45_output_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  tile_52_output, AI_STATIC,
  178, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 5, 50), AI_STRIDE_INIT(4, 4, 4, 40, 200),
  1, &tile_52_output_array, NULL)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  tile_52_output0, AI_STATIC,
  179, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &tile_52_output_array, NULL)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  tile_77_output, AI_STATIC,
  180, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 5, 250), AI_STRIDE_INIT(4, 4, 4, 32, 160),
  1, &tile_77_output_array, NULL)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  tile_77_output0, AI_STATIC,
  181, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &tile_77_output_array, NULL)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  transpose_16_output, AI_STATIC,
  182, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 50, 1), AI_STRIDE_INIT(4, 4, 4, 12, 600),
  1, &transpose_16_output_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  transpose_18_output, AI_STATIC,
  183, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 50, 3), AI_STRIDE_INIT(4, 4, 4, 4, 200),
  1, &transpose_18_output_array, NULL)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  transpose_18_output0, AI_STATIC,
  184, 0x0,
  AI_SHAPE_INIT(5, 1, 1, 50, 3, 1), AI_STRIDE_INIT(5, 4, 4, 4, 200, 4),
  1, &transpose_18_output_array, NULL)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  xs_0_Transpose_output, AI_STATIC,
  185, 0x0,
  AI_SHAPE_INIT(4, 1, 2500, 1, 3), AI_STRIDE_INIT(4, 4, 4, 10000, 10000),
  1, &xs_0_Transpose_output_array, NULL)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  xs_0_Transpose_output0, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 2500, 3), AI_STRIDE_INIT(4, 4, 4, 4, 10000),
  1, &xs_0_Transpose_output_array, NULL)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  xs_0_output, AI_STATIC,
  187, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 2500), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &xs_0_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_12_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_12_nl_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_12_nl_chain,
  NULL, &nl_12_nl_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_12_weights, &gemm_12_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_12_layer, 12,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_12_chain,
  NULL, &nl_12_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_11_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_11_layer, 11,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_11_chain,
  NULL, &gemm_12_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_9_weights, &conv2d_9_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_9_scratch0, &conv2d_9_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_9_layer, 10,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_9_chain,
  NULL, &pool_11_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .pool_size = AI_SHAPE_2D_INIT(4, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(4, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_7_scratch0, &conv2d_7_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 8,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_7_chain,
  NULL, &conv2d_9_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_5_weights, &conv2d_5_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_5_scratch0, &conv2d_5_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_5_layer, 6,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_5_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_86_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_83_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_86_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_86_layer, 86,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_86_chain,
  NULL, &conv2d_5_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_82_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_83_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_83_weights, &conv2d_83_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_83_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_83_layer, 85,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_83_chain,
  NULL, &nl_86_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_82_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_82_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_82_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_82_nl_layer, 82,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_82_nl_chain,
  NULL, &conv2d_83_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_82_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_81_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_82_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_82_weights, &conv2d_82_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_82_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_82_layer, 82,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_82_chain,
  NULL, &nl_82_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_81_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_81_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_81_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_81_nl_layer, 81,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_81_nl_chain,
  NULL, &conv2d_82_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_81_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_79_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_81_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_81_weights, &conv2d_81_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_81_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_81_layer, 81,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_81_chain,
  NULL, &nl_81_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_79_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tile_77_output0, &nl_30_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_79_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_79_layer, 79,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_79_chain,
  NULL, &conv2d_81_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_i16 tile_77_repeats_data[] = { 1, 5, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    tile_77_repeats, AI_ARRAY_FORMAT_S16,
    tile_77_repeats_data, tile_77_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  tile_77_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_75_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &tile_77_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  tile_77_layer, 77,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &tile_77_chain,
  NULL, &concat_79_layer, AI_STATIC, 
  .repeats = &tile_77_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_75_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_75_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_75_nl_layer, 75,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_75_nl_chain,
  NULL, &tile_77_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_75_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_74_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_75_weights, &conv2d_75_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_75_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_75_layer, 75,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_75_chain,
  NULL, &nl_75_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_74_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_74_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_74_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_74_nl_layer, 74,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_74_nl_chain,
  NULL, &conv2d_75_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_74_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_72_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_74_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_74_weights, &conv2d_74_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_74_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_74_layer, 74,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_74_chain,
  NULL, &nl_74_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_72_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &tile_52_output0, &nl_34_nl_output0, &nl_70_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_72_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_72_layer, 72,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_72_chain,
  NULL, &conv2d_74_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_i16 tile_52_repeats_data[] = { 1, 5, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    tile_52_repeats, AI_ARRAY_FORMAT_S16,
    tile_52_repeats_data, tile_52_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  tile_52_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_50_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &tile_52_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  tile_52_layer, 52,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &tile_52_chain,
  NULL, &concat_72_layer, AI_STATIC, 
  .repeats = &tile_52_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_50_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_50_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_50_nl_layer, 50,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_50_nl_chain,
  NULL, &tile_52_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_50_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_49_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_50_weights, &conv2d_50_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_50_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_50_layer, 50,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_50_chain,
  NULL, &nl_50_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_49_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_49_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_49_nl_layer, 49,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_49_nl_chain,
  NULL, &conv2d_50_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_47_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_49_weights, &conv2d_49_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_49_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_49_layer, 49,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_49_chain,
  NULL, &nl_49_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tile_45_output0, &nl_38_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_47_layer, 47,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_47_chain,
  NULL, &conv2d_49_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_i16 tile_45_repeats_data[] = { 1, 5, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    tile_45_repeats, AI_ARRAY_FORMAT_S16,
    tile_45_repeats_data, tile_45_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  tile_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_42_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &tile_45_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  tile_45_layer, 45,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &tile_45_chain,
  NULL, &concat_47_layer, AI_STATIC, 
  .repeats = &tile_45_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_42_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_42_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_42_nl_layer, 42,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_42_nl_chain,
  NULL, &tile_45_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_42_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_41_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_42_weights, &conv2d_42_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_42_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_42_layer, 42,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_42_chain,
  NULL, &nl_42_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_41_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_41_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_41_nl_layer, 41,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_41_nl_chain,
  NULL, &conv2d_42_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_41_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_40_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_41_weights, &conv2d_41_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_41_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_41_layer, 41,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_41_chain,
  NULL, &nl_41_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_38_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_40_layer, 40,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_40_chain,
  NULL, &conv2d_41_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_38_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_38_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_38_nl_layer, 38,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_38_nl_chain,
  NULL, &pool_40_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_37_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_38_weights, &conv2d_38_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_38_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_38_layer, 38,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_38_chain,
  NULL, &nl_38_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_37_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_37_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_37_nl_layer, 37,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_37_nl_chain,
  NULL, &conv2d_38_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_37_weights, &conv2d_37_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_37_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_37_layer, 37,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_37_chain,
  NULL, &nl_37_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_34_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_36_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_36_layer, 36,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_36_chain,
  NULL, &conv2d_37_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_70_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_70_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_70_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_70_nl_layer, 70,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_70_nl_chain,
  NULL, &pool_36_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_70_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_54_nl_output, &conv2d_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_70_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_70_layer, 70,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_70_chain,
  NULL, &nl_70_nl_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_69_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_67_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_69_weights, &conv2d_69_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_69_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_69_layer, 69,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_69_chain,
  NULL, &eltwise_70_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_67_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_65_output0, &eltwise_58_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_67_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_67_layer, 67,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_67_chain,
  NULL, &conv2d_69_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_58_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_55_output0, &nl_57_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_58_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_58_layer, 58,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_58_chain,
  NULL, &eltwise_67_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_57_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_55_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_57_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_57_layer, 57,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_57_chain,
  NULL, &eltwise_58_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_54_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_55_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_55_weights, &conv2d_59_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_55_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_55_layer, 55,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_55_chain,
  NULL, &nl_57_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_65_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_63_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_65_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_65_weights, &conv2d_59_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_65_layer, 65,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_65_chain,
  NULL, &conv2d_55_layer, AI_STATIC, 
  .groups = 10, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 127, 0, 127), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_63_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_60_output0, &nl_62_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_63_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_63_layer, 63,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_63_chain,
  NULL, &conv2d_65_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_62_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_60_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_62_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_62_layer, 62,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_62_chain,
  NULL, &eltwise_63_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_60_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_59_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_60_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_60_weights, &conv2d_60_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_60_layer, 60,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_60_chain,
  NULL, &nl_62_layer, AI_STATIC, 
  .groups = 10, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_59_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_54_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_59_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_59_weights, &conv2d_59_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_59_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_59_layer, 59,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_59_chain,
  NULL, &conv2d_60_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_54_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_54_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_54_nl_layer, 54,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_54_nl_chain,
  NULL, &conv2d_59_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_34_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_54_weights, &conv2d_54_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_54_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_54_layer, 54,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_54_chain,
  NULL, &nl_54_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_34_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_34_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_34_nl_layer, 34,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_34_nl_chain,
  NULL, &conv2d_54_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_33_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_34_weights, &conv2d_34_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_34_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_34_layer, 34,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_34_chain,
  NULL, &nl_34_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_33_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_33_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_33_nl_layer, 33,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_33_nl_chain,
  NULL, &conv2d_34_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_33_weights, &conv2d_33_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_33_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_33_layer, 33,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_33_chain,
  NULL, &nl_33_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_30_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_32_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_32_layer, 32,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_32_chain,
  NULL, &conv2d_33_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_30_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_30_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_30_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_30_nl_layer, 30,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_30_nl_chain,
  NULL, &pool_32_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_29_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_30_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_30_weights, &conv2d_30_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_30_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_30_layer, 30,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_30_chain,
  NULL, &nl_30_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_29_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_29_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_29_nl_layer, 29,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_29_nl_chain,
  NULL, &conv2d_30_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_29_weights, &conv2d_29_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_29_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_29_layer, 29,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_29_chain,
  NULL, &nl_29_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_26_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_28_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_28_layer, 28,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_28_chain,
  NULL, &conv2d_29_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &reduce_23_output0, &pool_24_output0, &pool_25_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_26_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_26_layer, 26,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_26_chain,
  NULL, &pool_28_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_float reduce_23_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_23_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_23_neutral_value_data, reduce_23_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_23_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_23_layer, 23,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_23_chain,
  NULL, &concat_26_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_23_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tile_20_output0, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_22_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_22_layer, 22,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_22_chain,
  NULL, &reduce_23_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)


AI_STATIC_CONST ai_i16 tile_20_repeats_data[] = { 1, 1, 1, 50, 1 };
AI_ARRAY_OBJ_DECLARE(
    tile_20_repeats, AI_ARRAY_FORMAT_S16,
    tile_20_repeats_data, tile_20_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  tile_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_18_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &tile_20_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  tile_20_layer, 20,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &tile_20_chain,
  NULL, &eltwise_22_layer, AI_STATIC, 
  .repeats = &tile_20_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_18_layer, 18,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_18_chain,
  NULL, &tile_20_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_17_layer, 17,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_17_chain,
  NULL, &transpose_18_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_16_layer, 16,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_16_chain,
  NULL, &nl_17_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_14_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_15_weights, &gemm_15_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_15_layer, 15,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_15_chain,
  NULL, &transpose_16_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_14_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_14_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_14_nl_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_14_nl_chain,
  NULL, &gemm_15_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_14_weights, &gemm_14_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_14_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_14_chain,
  NULL, &nl_14_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_13_layer, 13,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_13_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(50, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(50, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_24_layer, 24,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_24_chain,
  NULL, &pool_13_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 3), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_25_layer, 25,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_25_chain,
  NULL, &pool_24_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 3), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_4_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_4_nl_layer, 4,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_4_nl_chain,
  NULL, &pool_25_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_3_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_4_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_4_chain,
  NULL, &nl_4_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 4, 0, 4), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_3_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_3_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_3_nl_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_3_nl_chain,
  NULL, &conv2d_4_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &xs_0_Transpose_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_3_weights, &conv2d_3_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_3_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_layer, 3,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_3_chain,
  NULL, &nl_3_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 4, 0, 4), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  xs_0_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &xs_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &xs_0_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  xs_0_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &xs_0_Transpose_chain,
  NULL, &conv2d_3_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 42384, 1, 1),
    42384, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 360000, 1, 1),
    360000, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECGA_IN_NUM, &xs_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECGA_OUT_NUM, &conv2d_83_output0, &nl_86_output, &nl_30_nl_output0, &nl_34_nl_output0, &nl_38_nl_output0, &nl_42_nl_output0, &nl_70_nl_output0, &nl_12_nl_output),
  &xs_0_Transpose_layer, 0x7c4d4b36, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 42384, 1, 1),
      42384, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 360000, 1, 1),
      360000, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECGA_IN_NUM, &xs_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECGA_OUT_NUM, &conv2d_83_output0, &nl_86_output, &nl_30_nl_output0, &nl_34_nl_output0, &nl_38_nl_output0, &nl_42_nl_output0, &nl_70_nl_output0, &nl_12_nl_output),
  &xs_0_Transpose_layer, 0x7c4d4b36, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool ecga_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_ecga_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    xs_0_output_array.data = AI_PTR(g_ecga_activations_map[0] + 80144);
    xs_0_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 80144);
    xs_0_Transpose_output_array.data = AI_PTR(g_ecga_activations_map[0] + 50144);
    xs_0_Transpose_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 50144);
    conv2d_3_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 50108);
    conv2d_3_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 50108);
    conv2d_3_output_array.data = AI_PTR(g_ecga_activations_map[0] + 80144);
    conv2d_3_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 80144);
    nl_3_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 80144);
    nl_3_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 80144);
    conv2d_4_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 200144);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 200144);
    conv2d_4_output_array.data = AI_PTR(g_ecga_activations_map[0] + 40000);
    conv2d_4_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 40000);
    nl_4_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 40000);
    nl_4_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 40000);
    pool_25_output_array.data = AI_PTR(g_ecga_activations_map[0] + 160000);
    pool_25_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 160000);
    pool_24_output_array.data = AI_PTR(g_ecga_activations_map[0] + 200000);
    pool_24_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 200000);
    pool_13_output_array.data = AI_PTR(g_ecga_activations_map[0] + 37600);
    pool_13_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 37600);
    gemm_14_output_array.data = AI_PTR(g_ecga_activations_map[0] + 240000);
    gemm_14_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 240000);
    nl_14_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 240000);
    nl_14_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 240000);
    gemm_15_output_array.data = AI_PTR(g_ecga_activations_map[0] + 39400);
    gemm_15_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 39400);
    transpose_16_output_array.data = AI_PTR(g_ecga_activations_map[0] + 241800);
    transpose_16_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 241800);
    nl_17_output_array.data = AI_PTR(g_ecga_activations_map[0] + 241800);
    nl_17_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 241800);
    transpose_18_output_array.data = AI_PTR(g_ecga_activations_map[0] + 241200);
    transpose_18_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 241200);
    tile_20_output_array.data = AI_PTR(g_ecga_activations_map[0] + 10000);
    tile_20_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 10000);
    eltwise_22_output_array.data = AI_PTR(g_ecga_activations_map[0] + 240000);
    eltwise_22_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 240000);
    reduce_23_output_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    reduce_23_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    concat_26_output_array.data = AI_PTR(g_ecga_activations_map[0] + 240000);
    concat_26_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 240000);
    pool_28_output_array.data = AI_PTR(g_ecga_activations_map[0] + 160000);
    pool_28_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 160000);
    conv2d_29_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_29_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_29_output_array.data = AI_PTR(g_ecga_activations_map[0] + 336);
    conv2d_29_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 336);
    nl_29_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 160000);
    nl_29_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 160000);
    conv2d_30_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_30_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_30_output_array.data = AI_PTR(g_ecga_activations_map[0] + 168);
    conv2d_30_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 168);
    nl_30_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 160000);
    nl_30_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 160000);
    pool_32_output_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    pool_32_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_33_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 6000);
    conv2d_33_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 6000);
    conv2d_33_output_array.data = AI_PTR(g_ecga_activations_map[0] + 6120);
    conv2d_33_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 6120);
    nl_33_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 14120);
    nl_33_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 14120);
    conv2d_34_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_34_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_34_output_array.data = AI_PTR(g_ecga_activations_map[0] + 160);
    conv2d_34_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 160);
    nl_34_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 8160);
    nl_34_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 8160);
    conv2d_54_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_54_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_54_output_array.data = AI_PTR(g_ecga_activations_map[0] + 16160);
    conv2d_54_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 16160);
    nl_54_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 26160);
    nl_54_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 26160);
    conv2d_59_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_59_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_59_output_array.data = AI_PTR(g_ecga_activations_map[0] + 16160);
    conv2d_59_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 16160);
    conv2d_60_output_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    conv2d_60_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    nl_62_output_array.data = AI_PTR(g_ecga_activations_map[0] + 16160);
    nl_62_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 16160);
    eltwise_63_output_array.data = AI_PTR(g_ecga_activations_map[0] + 200000);
    eltwise_63_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 200000);
    conv2d_65_output_array.data = AI_PTR(g_ecga_activations_map[0] + 16160);
    conv2d_65_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 16160);
    conv2d_55_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_55_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_55_output_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    conv2d_55_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    nl_57_output_array.data = AI_PTR(g_ecga_activations_map[0] + 200000);
    nl_57_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 200000);
    eltwise_58_output_array.data = AI_PTR(g_ecga_activations_map[0] + 210000);
    eltwise_58_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 210000);
    eltwise_67_output_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    eltwise_67_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    conv2d_69_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_69_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_69_output_array.data = AI_PTR(g_ecga_activations_map[0] + 16160);
    conv2d_69_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 16160);
    eltwise_70_output_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    eltwise_70_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    nl_70_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 16160);
    nl_70_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 16160);
    pool_36_output_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    pool_36_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_37_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 1600);
    conv2d_37_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 1600);
    conv2d_37_output_array.data = AI_PTR(g_ecga_activations_map[0] + 1760);
    conv2d_37_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 1760);
    nl_37_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 3760);
    nl_37_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 3760);
    conv2d_38_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_38_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_38_output_array.data = AI_PTR(g_ecga_activations_map[0] + 200);
    conv2d_38_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 200);
    nl_38_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 2200);
    nl_38_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 2200);
    pool_40_output_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    pool_40_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_41_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 400);
    conv2d_41_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 400);
    conv2d_41_output_array.data = AI_PTR(g_ecga_activations_map[0] + 520);
    conv2d_41_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 520);
    nl_41_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    nl_41_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    conv2d_42_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_42_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_42_output_array.data = AI_PTR(g_ecga_activations_map[0] + 624);
    conv2d_42_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 624);
    nl_42_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 0);
    nl_42_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 0);
    tile_45_output_array.data = AI_PTR(g_ecga_activations_map[0] + 4200);
    tile_45_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 4200);
    concat_47_output_array.data = AI_PTR(g_ecga_activations_map[0] + 26160);
    concat_47_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 26160);
    conv2d_49_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_49_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_49_output_array.data = AI_PTR(g_ecga_activations_map[0] + 4200);
    conv2d_49_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 4200);
    nl_49_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 26160);
    nl_49_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 26160);
    conv2d_50_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_50_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_50_output_array.data = AI_PTR(g_ecga_activations_map[0] + 4200);
    conv2d_50_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 4200);
    nl_50_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 26160);
    nl_50_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 26160);
    tile_52_output_array.data = AI_PTR(g_ecga_activations_map[0] + 28160);
    tile_52_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 28160);
    concat_72_output_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    concat_72_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    conv2d_74_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_74_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_74_output_array.data = AI_PTR(g_ecga_activations_map[0] + 26160);
    conv2d_74_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 26160);
    nl_74_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    nl_74_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    conv2d_75_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_75_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_75_output_array.data = AI_PTR(g_ecga_activations_map[0] + 26160);
    conv2d_75_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 26160);
    nl_75_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    nl_75_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    tile_77_output_array.data = AI_PTR(g_ecga_activations_map[0] + 198000);
    tile_77_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 198000);
    concat_79_output_array.data = AI_PTR(g_ecga_activations_map[0] + 238000);
    concat_79_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 238000);
    conv2d_81_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_81_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_81_output_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    conv2d_81_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    nl_81_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 220000);
    nl_81_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 220000);
    conv2d_82_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_82_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_82_output_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    conv2d_82_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    nl_82_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 220000);
    nl_82_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 220000);
    conv2d_83_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_83_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_83_output_array.data = AI_PTR(g_ecga_activations_map[0] + 26160);
    conv2d_83_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 26160);
    nl_86_output_array.data = AI_PTR(g_ecga_activations_map[0] + 31160);
    nl_86_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 31160);
    conv2d_5_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_5_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_5_scratch1_array.data = AI_PTR(g_ecga_activations_map[0] + 190000);
    conv2d_5_scratch1_array.data_start = AI_PTR(g_ecga_activations_map[0] + 190000);
    conv2d_5_output_array.data = AI_PTR(g_ecga_activations_map[0] + 230000);
    conv2d_5_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 230000);
    conv2d_7_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_7_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_7_scratch1_array.data = AI_PTR(g_ecga_activations_map[0] + 36160);
    conv2d_7_scratch1_array.data_start = AI_PTR(g_ecga_activations_map[0] + 36160);
    conv2d_7_output_array.data = AI_PTR(g_ecga_activations_map[0] + 44160);
    conv2d_7_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 44160);
    conv2d_9_scratch0_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_9_scratch0_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    conv2d_9_scratch1_array.data = AI_PTR(g_ecga_activations_map[0] + 592);
    conv2d_9_scratch1_array.data_start = AI_PTR(g_ecga_activations_map[0] + 592);
    conv2d_9_output_array.data = AI_PTR(g_ecga_activations_map[0] + 4200);
    conv2d_9_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 4200);
    pool_11_output_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    pool_11_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    gemm_12_output_array.data = AI_PTR(g_ecga_activations_map[0] + 720);
    gemm_12_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 720);
    nl_12_nl_output_array.data = AI_PTR(g_ecga_activations_map[0] + 480);
    nl_12_nl_output_array.data_start = AI_PTR(g_ecga_activations_map[0] + 480);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool ecga_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_ecga_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 0);
    conv2d_3_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 0);
    conv2d_3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 144);
    conv2d_3_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 144);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 160);
    conv2d_4_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 160);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 736);
    conv2d_4_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 736);
    gemm_14_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_14_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 752);
    gemm_14_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 752);
    gemm_14_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_14_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 816);
    gemm_14_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 816);
    gemm_15_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_15_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 832);
    gemm_15_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 832);
    gemm_15_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_15_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 848);
    gemm_15_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 848);
    conv2d_29_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_29_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 852);
    conv2d_29_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 852);
    conv2d_29_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_29_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 2868);
    conv2d_29_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 2868);
    conv2d_30_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_30_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 2892);
    conv2d_30_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 2892);
    conv2d_30_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_30_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 3900);
    conv2d_30_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 3900);
    conv2d_33_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_33_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 3924);
    conv2d_33_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 3924);
    conv2d_33_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_33_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 4884);
    conv2d_33_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 4884);
    conv2d_34_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_34_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 4916);
    conv2d_34_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 4916);
    conv2d_34_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_34_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 6196);
    conv2d_34_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 6196);
    conv2d_54_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_54_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 6228);
    conv2d_54_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 6228);
    conv2d_54_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_54_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 6548);
    conv2d_54_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 6548);
    conv2d_59_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_59_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 6588);
    conv2d_59_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 6588);
    conv2d_59_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_59_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 6988);
    conv2d_59_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 6988);
    conv2d_60_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_60_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 7028);
    conv2d_60_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 7028);
    conv2d_60_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_60_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 7148);
    conv2d_60_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 7148);
    conv2d_65_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_65_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 7188);
    conv2d_65_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 7188);
    conv2d_55_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_55_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 17388);
    conv2d_55_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 17388);
    conv2d_69_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_69_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 17788);
    conv2d_69_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 17788);
    conv2d_69_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_69_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 18188);
    conv2d_69_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 18188);
    conv2d_37_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_37_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 18228);
    conv2d_37_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 18228);
    conv2d_37_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_37_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 19828);
    conv2d_37_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 19828);
    conv2d_38_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_38_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 19868);
    conv2d_38_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 19868);
    conv2d_38_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_38_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 21868);
    conv2d_38_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 21868);
    conv2d_41_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_41_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 21908);
    conv2d_41_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 21908);
    conv2d_41_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_41_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 23348);
    conv2d_41_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 23348);
    conv2d_42_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_42_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 23396);
    conv2d_42_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 23396);
    conv2d_42_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_42_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 25124);
    conv2d_42_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 25124);
    conv2d_49_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_49_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 25172);
    conv2d_49_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 25172);
    conv2d_49_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_49_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 29572);
    conv2d_49_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 29572);
    conv2d_50_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_50_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 29612);
    conv2d_50_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 29612);
    conv2d_50_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_50_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 31612);
    conv2d_50_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 31612);
    conv2d_74_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_74_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 31652);
    conv2d_74_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 31652);
    conv2d_74_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_74_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 36132);
    conv2d_74_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 36132);
    conv2d_75_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_75_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 36164);
    conv2d_75_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 36164);
    conv2d_75_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_75_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 37444);
    conv2d_75_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 37444);
    conv2d_81_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_81_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 37476);
    conv2d_81_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 37476);
    conv2d_81_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_81_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 39828);
    conv2d_81_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 39828);
    conv2d_82_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_82_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 39852);
    conv2d_82_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 39852);
    conv2d_82_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_82_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 40860);
    conv2d_82_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 40860);
    conv2d_83_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_83_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 40884);
    conv2d_83_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 40884);
    conv2d_83_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_83_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 40908);
    conv2d_83_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 40908);
    conv2d_5_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 40912);
    conv2d_5_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 40912);
    conv2d_5_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 41360);
    conv2d_5_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 41360);
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 41376);
    conv2d_7_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 41376);
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 41824);
    conv2d_7_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 41824);
    conv2d_9_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 41840);
    conv2d_9_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 41840);
    conv2d_9_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 42288);
    conv2d_9_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 42288);
    gemm_12_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_12_weights_array.data = AI_PTR(g_ecga_weights_map[0] + 42304);
    gemm_12_weights_array.data_start = AI_PTR(g_ecga_weights_map[0] + 42304);
    gemm_12_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_12_bias_array.data = AI_PTR(g_ecga_weights_map[0] + 42368);
    gemm_12_bias_array.data_start = AI_PTR(g_ecga_weights_map[0] + 42368);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_ecga_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_ECGA_MODEL_NAME,
      .model_signature   = AI_ECGA_MODEL_SIGNATURE,
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
      
      .n_macc            = 6198021,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x7c4d4b36,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_ecga_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_ECGA_MODEL_NAME,
      .model_signature   = AI_ECGA_MODEL_SIGNATURE,
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
      
      .n_macc            = 6198021,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x7c4d4b36,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_ecga_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_ecga_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_ecga_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_ecga_create(network, AI_ECGA_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_ecga_data_params_get(&params) != true) {
    err = ai_ecga_get_error(*network);
    return err;
  }
#if defined(AI_ECGA_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_ECGA_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_ecga_init(*network, &params) != true) {
    err = ai_ecga_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_ecga_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_ecga_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_ecga_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_ecga_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= ecga_configure_weights(net_ctx, params);
  ok &= ecga_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_ecga_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_ecga_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_ECGA_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

