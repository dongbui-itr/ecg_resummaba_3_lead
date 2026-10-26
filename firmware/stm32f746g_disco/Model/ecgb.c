/**
  ******************************************************************************
  * @file    ecgb.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-25T22:15:55+0700
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


#include "ecgb.h"
#include "ecgb_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_ecgb
 
#undef AI_ECGB_MODEL_SIGNATURE
#define AI_ECGB_MODEL_SIGNATURE     "0xb1d55ea02fad3912010f6b6e05863d82"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-10-25T22:15:55+0700"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_ECGB_N_BATCHES
#define AI_ECGB_N_BATCHES         (1)

static ai_ptr g_ecgb_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_ecgb_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  xs_9_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 60, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gemm_146_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  transpose_148_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  pool_141_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  pool_139_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  xs_4_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 7500, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  xs_8_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2500, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  xs_5_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2000, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  xs_6_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 500, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  xs_7_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 120, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  xs_3_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1250, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  nl_54_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1250, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_55_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1250, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  xs_2_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 216, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  gemm_48_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  nl_48_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  gemm_49_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  split_52_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  split_52_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_53_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  xs_1_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1250, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  xs_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3750, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  xs_0_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3750, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15000, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  nl_3_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15000, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15000, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  nl_4_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15000, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  pool_17_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  pool_16_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  pool_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 600, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 600, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  nl_6_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 600, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  transpose_8_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  nl_9_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  transpose_10_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  tile_12_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3750, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_14_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15000, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  reduce_15_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  concat_18_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15000, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  concat_19_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16250, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  nl_21_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  nl_22_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_23_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  nl_24_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_25_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  nl_26_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_27_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  nl_28_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_29_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5000, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  pool_31_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1000, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  nl_32_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  nl_33_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_34_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  nl_35_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_36_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  nl_37_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_38_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1500, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  pool_40_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 300, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 400, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  nl_41_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 400, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_42_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 400, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  nl_42_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 400, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  pool_44_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 80, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_45_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  nl_45_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_46_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  nl_46_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  concat_56_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 220, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  tile_58_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1100, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  concat_60_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_62_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  nl_62_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_63_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  nl_63_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  tile_65_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  concat_67_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8500, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_69_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  nl_69_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_70_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  nl_70_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_75_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_76_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  nl_78_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_79_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_81_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_71_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  nl_73_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_74_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_83_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_85_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_86_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  nl_86_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  tile_91_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12500, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  concat_93_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25000, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_95_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  nl_95_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_96_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  nl_96_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7500, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_97_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3750, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  nl_99_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3750, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_100_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3750, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  concat_101_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 5000, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  pool_103_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1000, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  concat_105_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3500, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_107_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  nl_107_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_109_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_110_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_116_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_117_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  nl_119_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_120_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_122_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_112_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  nl_114_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_115_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_124_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_126_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_128_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  nl_128_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2500, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  tile_130_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12500, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  concat_132_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25000, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_134_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10000, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  nl_134_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10000, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_135_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10000, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  nl_135_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10000, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_136_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6250, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  nl_138_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6250, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  pool_88_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  concat_143_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 90, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  gemm_144_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  nl_144_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  gemm_145_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  concat_149_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  nl_150_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 20, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  arith_constant116_3D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  gemm_146_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  gemm_146_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  gemm_48_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2160, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  gemm_48_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  gemm_49_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 200, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  gemm_49_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  split_52_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_53_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_53_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 28, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 52, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 180, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 180, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 180, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 240, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_42_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 320, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_42_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_45_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 240, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_45_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_46_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 300, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_46_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_62_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_62_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_63_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_63_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_69_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1700, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_69_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_70_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 500, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_70_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_75_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_75_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_76_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_76_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_81_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2550, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_71_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_85_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_85_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_95_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 840, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_95_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#207 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_96_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 252, AI_STATIC)

/* Array#208 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_96_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#209 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_97_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18, AI_STATIC)

/* Array#210 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_97_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#211 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_107_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 140, AI_STATIC)

/* Array#212 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_107_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#213 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_116_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#214 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_117_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)

/* Array#215 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_117_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#216 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_122_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2550, AI_STATIC)

/* Array#217 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_112_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#218 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_126_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#219 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_126_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#220 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_134_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1120, AI_STATIC)

/* Array#221 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_134_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#222 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_135_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 448, AI_STATIC)

/* Array#223 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_135_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#224 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_136_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40, AI_STATIC)

/* Array#225 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_136_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)

/* Array#226 */
AI_ARRAY_OBJ_DECLARE(
  gemm_144_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 72, AI_STATIC)

/* Array#227 */
AI_ARRAY_OBJ_DECLARE(
  gemm_144_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#228 */
AI_ARRAY_OBJ_DECLARE(
  gemm_145_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#229 */
AI_ARRAY_OBJ_DECLARE(
  gemm_145_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#230 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#231 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 28, AI_STATIC)

/* Array#232 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 13, AI_STATIC)

/* Array#233 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 28, AI_STATIC)

/* Array#234 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)

/* Array#235 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)

/* Array#236 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)

/* Array#237 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_42_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40, AI_STATIC)

/* Array#238 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_45_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#239 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_46_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)

/* Array#240 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_62_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 200, AI_STATIC)

/* Array#241 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_63_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#242 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_69_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 170, AI_STATIC)

/* Array#243 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_70_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#244 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_75_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#245 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_71_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#246 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_85_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#247 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_95_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 140, AI_STATIC)

/* Array#248 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_96_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 42, AI_STATIC)

/* Array#249 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_97_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#250 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_107_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 14, AI_STATIC)

/* Array#251 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_116_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#252 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_112_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#253 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_126_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)

/* Array#254 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_134_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 140, AI_STATIC)

/* Array#255 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_135_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#256 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_136_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  arith_constant116_3D, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &arith_constant116_3D_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  concat_101_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &concat_101_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  concat_101_output0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &concat_101_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  concat_105_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 14, 1, 250), AI_STRIDE_INIT(4, 4, 4, 56, 56),
  1, &concat_105_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  concat_105_output0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 14, 250, 1), AI_STRIDE_INIT(4, 4, 4, 56, 14000),
  1, &concat_105_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  concat_132_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 80, 80),
  1, &concat_132_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  concat_132_output0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 80, 100000),
  1, &concat_132_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  concat_143_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 18, 1, 5), AI_STRIDE_INIT(4, 4, 4, 72, 72),
  1, &concat_143_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  concat_149_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 5), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &concat_149_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  concat_18_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &concat_18_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  concat_19_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 13, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 52, 52),
  1, &concat_19_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  concat_19_output0, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 13, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 52, 65000),
  1, &concat_19_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  concat_56_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 1, 10), AI_STRIDE_INIT(4, 4, 4, 88, 88),
  1, &concat_56_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  concat_60_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 40, 1, 50), AI_STRIDE_INIT(4, 4, 4, 160, 160),
  1, &concat_60_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  concat_60_output0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 40, 50, 1), AI_STRIDE_INIT(4, 4, 4, 160, 8000),
  1, &concat_60_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  concat_67_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 1, 250), AI_STRIDE_INIT(4, 4, 4, 136, 136),
  1, &concat_67_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  concat_67_output0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 250, 1), AI_STRIDE_INIT(4, 4, 4, 136, 34000),
  1, &concat_67_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  concat_93_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 80, 80),
  1, &concat_93_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  concat_93_output0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 80, 100000),
  1, &concat_93_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_107_bias, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_107_bias_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_107_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_107_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_107_scratch0, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 14, 1, 1), AI_STRIDE_INIT(4, 4, 4, 56, 56),
  1, &conv2d_107_scratch0_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_107_weights, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 14, 1, 1, 10), AI_STRIDE_INIT(4, 4, 56, 560, 560),
  1, &conv2d_107_weights_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_112_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_112_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_112_output0, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_112_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_112_scratch0, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_112_scratch0_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_112_weights, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 10, 1, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 400),
  1, &conv2d_112_weights_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_116_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_116_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_116_scratch0, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_116_scratch0_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_116_weights, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 10, 1, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 400),
  1, &conv2d_116_weights_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_117_bias, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_117_bias_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_117_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_117_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_117_output0, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_117_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_117_weights, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 10), AI_STRIDE_INIT(4, 1, 10, 10, 10),
  1, &conv2d_117_weights_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_122_output, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_122_output_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_122_output0, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_122_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_122_weights, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 255, 1, 10), AI_STRIDE_INIT(4, 1, 10, 10, 10),
  1, &conv2d_122_weights_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_126_bias, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_126_bias_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_126_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_126_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_126_output0, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_126_output_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_126_scratch0, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_126_scratch0_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_126_weights, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 10, 1, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 400),
  1, &conv2d_126_weights_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_134_bias, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_134_bias_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_134_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 40000),
  1, &conv2d_134_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_134_scratch0, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 7, 1), AI_STRIDE_INIT(4, 4, 4, 80, 560),
  1, &conv2d_134_scratch0_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_134_weights, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 20, 7, 1, 8), AI_STRIDE_INIT(4, 4, 80, 640, 4480),
  1, &conv2d_134_weights_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_135_bias, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_135_bias_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_135_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 40000),
  1, &conv2d_135_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_135_scratch0, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 7, 1), AI_STRIDE_INIT(4, 4, 4, 32, 224),
  1, &conv2d_135_scratch0_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_135_weights, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 8, 7, 1, 8), AI_STRIDE_INIT(4, 4, 32, 256, 1792),
  1, &conv2d_135_weights_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_136_bias, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &conv2d_136_bias_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_136_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 20, 25000),
  1, &conv2d_136_output_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_136_output0, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &conv2d_136_output_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_136_scratch0, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_136_scratch0_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_136_weights, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 5), AI_STRIDE_INIT(4, 4, 32, 160, 160),
  1, &conv2d_136_weights_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_bias, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_21_bias_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_output, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &conv2d_21_output_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_scratch0, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 13, 1, 1), AI_STRIDE_INIT(4, 4, 4, 52, 52),
  1, &conv2d_21_scratch0_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_weights, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 13, 1, 1, 4), AI_STRIDE_INIT(4, 4, 52, 208, 208),
  1, &conv2d_21_weights_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_bias, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_22_bias_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_output, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &conv2d_22_output_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_scratch0, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 7, 1), AI_STRIDE_INIT(4, 4, 4, 16, 112),
  1, &conv2d_22_scratch0_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_weights, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 4, 7, 1, 4), AI_STRIDE_INIT(4, 4, 16, 64, 448),
  1, &conv2d_22_weights_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_bias, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_24_bias_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_output, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &conv2d_24_output_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_weights, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 4, 7, 1, 4), AI_STRIDE_INIT(4, 4, 16, 64, 448),
  1, &conv2d_24_weights_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_bias, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_26_bias_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_output, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &conv2d_26_output_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_weights, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 4, 7, 1, 4), AI_STRIDE_INIT(4, 4, 16, 64, 448),
  1, &conv2d_26_weights_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_bias, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_28_bias_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_output, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &conv2d_28_output_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_weights, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 4, 7, 1, 4), AI_STRIDE_INIT(4, 4, 16, 64, 448),
  1, &conv2d_28_weights_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_bias, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_32_bias_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_output, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &conv2d_32_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_scratch0, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_32_scratch0_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_weights, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 4, 1, 1, 6), AI_STRIDE_INIT(4, 4, 16, 96, 96),
  1, &conv2d_32_weights_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_bias, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_33_bias_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &conv2d_33_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_scratch0, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 5, 1), AI_STRIDE_INIT(4, 4, 4, 24, 120),
  1, &conv2d_33_scratch0_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_weights, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 6, 5, 1, 6), AI_STRIDE_INIT(4, 4, 24, 144, 720),
  1, &conv2d_33_weights_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_bias, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_35_bias_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_output, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &conv2d_35_output_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_weights, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 6, 5, 1, 6), AI_STRIDE_INIT(4, 4, 24, 144, 720),
  1, &conv2d_35_weights_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_bias, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_37_bias_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &conv2d_37_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_weights, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 6, 5, 1, 6), AI_STRIDE_INIT(4, 4, 24, 144, 720),
  1, &conv2d_37_weights_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_bias, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_3_bias_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 3), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &conv2d_3_output_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_scratch0, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 7, 1), AI_STRIDE_INIT(4, 4, 4, 4, 28),
  1, &conv2d_3_scratch0_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_weights, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 4), AI_STRIDE_INIT(4, 4, 4, 16, 112),
  1, &conv2d_3_weights_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_bias, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_41_bias_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_output, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 50, 1), AI_STRIDE_INIT(4, 4, 4, 32, 1600),
  1, &conv2d_41_output_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_scratch0, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 5, 1), AI_STRIDE_INIT(4, 4, 4, 24, 120),
  1, &conv2d_41_scratch0_array, NULL)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_weights, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 6, 5, 1, 8), AI_STRIDE_INIT(4, 4, 24, 192, 960),
  1, &conv2d_41_weights_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_42_bias, AI_STATIC,
  94, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_42_bias_array, NULL)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_42_output, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 50, 1), AI_STRIDE_INIT(4, 4, 4, 32, 1600),
  1, &conv2d_42_output_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_42_scratch0, AI_STATIC,
  96, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 5, 1), AI_STRIDE_INIT(4, 4, 4, 32, 160),
  1, &conv2d_42_scratch0_array, NULL)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_42_weights, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 8, 5, 1, 8), AI_STRIDE_INIT(4, 4, 32, 256, 1280),
  1, &conv2d_42_weights_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_45_bias, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_45_bias_array, NULL)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_45_output, AI_STATIC,
  99, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 10, 1), AI_STRIDE_INIT(4, 4, 4, 40, 400),
  1, &conv2d_45_output_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_45_scratch0, AI_STATIC,
  100, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 3, 1), AI_STRIDE_INIT(4, 4, 4, 32, 96),
  1, &conv2d_45_scratch0_array, NULL)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_45_weights, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 8, 3, 1, 10), AI_STRIDE_INIT(4, 4, 32, 320, 960),
  1, &conv2d_45_weights_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_46_bias, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_46_bias_array, NULL)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_46_output, AI_STATIC,
  103, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 10, 1), AI_STRIDE_INIT(4, 4, 4, 40, 400),
  1, &conv2d_46_output_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_46_scratch0, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 3, 1), AI_STRIDE_INIT(4, 4, 4, 40, 120),
  1, &conv2d_46_scratch0_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_46_weights, AI_STATIC,
  105, 0x0,
  AI_SHAPE_INIT(4, 10, 3, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 1200),
  1, &conv2d_46_weights_array, NULL)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  107, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 3), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &conv2d_4_output_array, NULL)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  108, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 7, 1), AI_STRIDE_INIT(4, 4, 4, 16, 112),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  109, 0x0,
  AI_SHAPE_INIT(4, 4, 7, 1, 4), AI_STRIDE_INIT(4, 4, 16, 64, 448),
  1, &conv2d_4_weights_array, NULL)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_62_bias, AI_STATIC,
  110, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_62_bias_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_62_output, AI_STATIC,
  111, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &conv2d_62_output_array, NULL)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_62_scratch0, AI_STATIC,
  112, 0x0,
  AI_SHAPE_INIT(4, 1, 40, 5, 1), AI_STRIDE_INIT(4, 4, 4, 160, 800),
  1, &conv2d_62_scratch0_array, NULL)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_62_weights, AI_STATIC,
  113, 0x0,
  AI_SHAPE_INIT(4, 40, 5, 1, 10), AI_STRIDE_INIT(4, 4, 160, 1600, 8000),
  1, &conv2d_62_weights_array, NULL)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_63_bias, AI_STATIC,
  114, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_63_bias_array, NULL)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_63_output, AI_STATIC,
  115, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &conv2d_63_output_array, NULL)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_63_scratch0, AI_STATIC,
  116, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 5, 1), AI_STRIDE_INIT(4, 4, 4, 40, 200),
  1, &conv2d_63_scratch0_array, NULL)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_63_weights, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 10, 5, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 2000),
  1, &conv2d_63_weights_array, NULL)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_69_bias, AI_STATIC,
  118, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_69_bias_array, NULL)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_69_output, AI_STATIC,
  119, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_69_output_array, NULL)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_69_scratch0, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 1, 34, 5, 1), AI_STRIDE_INIT(4, 4, 4, 136, 680),
  1, &conv2d_69_scratch0_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_69_weights, AI_STATIC,
  121, 0x0,
  AI_SHAPE_INIT(4, 34, 5, 1, 10), AI_STRIDE_INIT(4, 4, 136, 1360, 6800),
  1, &conv2d_69_weights_array, NULL)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_70_bias, AI_STATIC,
  122, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_70_bias_array, NULL)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_70_output, AI_STATIC,
  123, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_70_output_array, NULL)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_70_scratch0, AI_STATIC,
  124, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 5, 1), AI_STRIDE_INIT(4, 4, 4, 40, 200),
  1, &conv2d_70_scratch0_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_70_weights, AI_STATIC,
  125, 0x0,
  AI_SHAPE_INIT(4, 10, 5, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 2000),
  1, &conv2d_70_weights_array, NULL)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_71_output, AI_STATIC,
  126, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_71_output_array, NULL)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_71_output0, AI_STATIC,
  127, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_71_output_array, NULL)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_71_scratch0, AI_STATIC,
  128, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_71_scratch0_array, NULL)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_71_weights, AI_STATIC,
  129, 0x0,
  AI_SHAPE_INIT(4, 10, 1, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 400),
  1, &conv2d_71_weights_array, NULL)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_75_bias, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_75_bias_array, NULL)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_75_output, AI_STATIC,
  131, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_75_output_array, NULL)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_75_scratch0, AI_STATIC,
  132, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_75_scratch0_array, NULL)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_75_weights, AI_STATIC,
  133, 0x0,
  AI_SHAPE_INIT(4, 10, 1, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 400),
  1, &conv2d_75_weights_array, NULL)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_76_bias, AI_STATIC,
  134, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_76_bias_array, NULL)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_76_output, AI_STATIC,
  135, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_76_output_array, NULL)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_76_output0, AI_STATIC,
  136, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_76_output_array, NULL)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_76_weights, AI_STATIC,
  137, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 10), AI_STRIDE_INIT(4, 1, 10, 10, 10),
  1, &conv2d_76_weights_array, NULL)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_81_output, AI_STATIC,
  138, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_81_output_array, NULL)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_81_output0, AI_STATIC,
  139, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_81_output_array, NULL)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_81_weights, AI_STATIC,
  140, 0x0,
  AI_SHAPE_INIT(4, 1, 255, 1, 10), AI_STRIDE_INIT(4, 1, 10, 10, 10),
  1, &conv2d_81_weights_array, NULL)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_85_bias, AI_STATIC,
  141, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_85_bias_array, NULL)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_85_output, AI_STATIC,
  142, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &conv2d_85_output_array, NULL)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_85_scratch0, AI_STATIC,
  143, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_85_scratch0_array, NULL)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_85_weights, AI_STATIC,
  144, 0x0,
  AI_SHAPE_INIT(4, 10, 1, 1, 10), AI_STRIDE_INIT(4, 4, 40, 400, 400),
  1, &conv2d_85_weights_array, NULL)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_95_bias, AI_STATIC,
  145, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_95_bias_array, NULL)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_95_output, AI_STATIC,
  146, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &conv2d_95_output_array, NULL)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_95_scratch0, AI_STATIC,
  147, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 7, 1), AI_STRIDE_INIT(4, 4, 4, 80, 560),
  1, &conv2d_95_scratch0_array, NULL)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_95_weights, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 20, 7, 1, 6), AI_STRIDE_INIT(4, 4, 80, 480, 3360),
  1, &conv2d_95_weights_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_96_bias, AI_STATIC,
  149, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_96_bias_array, NULL)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_96_output, AI_STATIC,
  150, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &conv2d_96_output_array, NULL)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_96_scratch0, AI_STATIC,
  151, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 7, 1), AI_STRIDE_INIT(4, 4, 4, 24, 168),
  1, &conv2d_96_scratch0_array, NULL)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_96_weights, AI_STATIC,
  152, 0x0,
  AI_SHAPE_INIT(4, 6, 7, 1, 6), AI_STRIDE_INIT(4, 4, 24, 144, 1008),
  1, &conv2d_96_weights_array, NULL)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_97_bias, AI_STATIC,
  153, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &conv2d_97_bias_array, NULL)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_97_output, AI_STATIC,
  154, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 12, 15000),
  1, &conv2d_97_output_array, NULL)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_97_output0, AI_STATIC,
  155, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &conv2d_97_output_array, NULL)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_97_scratch0, AI_STATIC,
  156, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &conv2d_97_scratch0_array, NULL)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_97_weights, AI_STATIC,
  157, 0x0,
  AI_SHAPE_INIT(4, 6, 1, 1, 3), AI_STRIDE_INIT(4, 4, 24, 72, 72),
  1, &conv2d_97_weights_array, NULL)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_100_output, AI_STATIC,
  158, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &eltwise_100_output_array, NULL)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_109_output, AI_STATIC,
  159, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_109_output_array, NULL)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_110_output, AI_STATIC,
  160, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_110_output_array, NULL)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_110_output0, AI_STATIC,
  161, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &eltwise_110_output_array, NULL)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_115_output, AI_STATIC,
  162, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_115_output_array, NULL)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_120_output, AI_STATIC,
  163, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_120_output_array, NULL)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_120_output0, AI_STATIC,
  164, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &eltwise_120_output_array, NULL)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_124_output, AI_STATIC,
  165, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_124_output_array, NULL)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_124_output0, AI_STATIC,
  166, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &eltwise_124_output_array, NULL)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_128_output, AI_STATIC,
  167, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_128_output_array, NULL)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_14_output, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 3), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &eltwise_14_output_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_23_output, AI_STATIC,
  169, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &eltwise_23_output_array, NULL)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_25_output, AI_STATIC,
  170, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &eltwise_25_output_array, NULL)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_27_output, AI_STATIC,
  171, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &eltwise_27_output_array, NULL)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_29_output, AI_STATIC,
  172, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &eltwise_29_output_array, NULL)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_29_output0, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &eltwise_29_output_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_34_output, AI_STATIC,
  174, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &eltwise_34_output_array, NULL)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_36_output, AI_STATIC,
  175, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &eltwise_36_output_array, NULL)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_38_output, AI_STATIC,
  176, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &eltwise_38_output_array, NULL)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_38_output0, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 250), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &eltwise_38_output_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_53_bias, AI_STATIC,
  178, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_53_bias_array, NULL)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_53_output, AI_STATIC,
  179, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_53_output_array, NULL)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_53_scale, AI_STATIC,
  180, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_53_scale_array, NULL)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_55_output, AI_STATIC,
  181, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &eltwise_55_output_array, NULL)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_74_output, AI_STATIC,
  182, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_74_output_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_79_output, AI_STATIC,
  183, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_79_output_array, NULL)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_79_output0, AI_STATIC,
  184, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &eltwise_79_output_array, NULL)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_83_output, AI_STATIC,
  185, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &eltwise_83_output_array, NULL)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_83_output0, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &eltwise_83_output_array, NULL)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_86_output, AI_STATIC,
  187, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &eltwise_86_output_array, NULL)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  gemm_144_bias, AI_STATIC,
  188, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_144_bias_array, NULL)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  gemm_144_output, AI_STATIC,
  189, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 5), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_144_output_array, NULL)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  gemm_144_weights, AI_STATIC,
  190, 0x0,
  AI_SHAPE_INIT(4, 18, 4, 1, 1), AI_STRIDE_INIT(4, 4, 72, 288, 288),
  1, &gemm_144_weights_array, NULL)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  gemm_145_bias, AI_STATIC,
  191, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_145_bias_array, NULL)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  gemm_145_output, AI_STATIC,
  192, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 5), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_145_output_array, NULL)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  gemm_145_weights, AI_STATIC,
  193, 0x0,
  AI_SHAPE_INIT(4, 4, 1, 1, 1), AI_STRIDE_INIT(4, 4, 16, 16, 16),
  1, &gemm_145_weights_array, NULL)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  gemm_146_bias, AI_STATIC,
  194, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_146_bias_array, NULL)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  gemm_146_output, AI_STATIC,
  195, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 3), AI_STRIDE_INIT(4, 4, 4, 4, 20),
  1, &gemm_146_output_array, NULL)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  gemm_146_output0, AI_STATIC,
  196, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 3), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &gemm_146_output_array, NULL)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  gemm_146_weights, AI_STATIC,
  197, 0x0,
  AI_SHAPE_INIT(4, 4, 1, 1, 1), AI_STRIDE_INIT(4, 4, 16, 16, 16),
  1, &gemm_146_weights_array, NULL)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_bias, AI_STATIC,
  198, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &gemm_48_bias_array, NULL)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_output, AI_STATIC,
  199, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &gemm_48_output_array, NULL)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  gemm_48_weights, AI_STATIC,
  200, 0x0,
  AI_SHAPE_INIT(4, 216, 10, 1, 1), AI_STRIDE_INIT(4, 4, 864, 8640, 8640),
  1, &gemm_48_weights_array, NULL)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_bias, AI_STATIC,
  201, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 1, 1), AI_STRIDE_INIT(4, 4, 4, 80, 80),
  1, &gemm_49_bias_array, NULL)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_output, AI_STATIC,
  202, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 1, 1), AI_STRIDE_INIT(4, 4, 4, 80, 80),
  1, &gemm_49_output_array, NULL)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  gemm_49_weights, AI_STATIC,
  203, 0x0,
  AI_SHAPE_INIT(4, 10, 20, 1, 1), AI_STRIDE_INIT(4, 4, 40, 800, 800),
  1, &gemm_49_weights_array, NULL)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_bias, AI_STATIC,
  204, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_6_bias_array, NULL)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_output, AI_STATIC,
  205, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 50, 3), AI_STRIDE_INIT(4, 4, 4, 16, 800),
  1, &gemm_6_output_array, NULL)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_weights, AI_STATIC,
  206, 0x0,
  AI_SHAPE_INIT(4, 4, 4, 1, 1), AI_STRIDE_INIT(4, 4, 16, 64, 64),
  1, &gemm_6_weights_array, NULL)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_bias, AI_STATIC,
  207, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_7_bias_array, NULL)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_output, AI_STATIC,
  208, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 50, 3), AI_STRIDE_INIT(4, 4, 4, 4, 200),
  1, &gemm_7_output_array, NULL)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_weights, AI_STATIC,
  209, 0x0,
  AI_SHAPE_INIT(4, 4, 1, 1, 1), AI_STRIDE_INIT(4, 4, 16, 16, 16),
  1, &gemm_7_weights_array, NULL)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  nl_107_nl_output, AI_STATIC,
  210, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &nl_107_nl_output_array, NULL)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  nl_107_nl_output0, AI_STATIC,
  211, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_107_nl_output_array, NULL)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  nl_114_output, AI_STATIC,
  212, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_114_output_array, NULL)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  nl_119_output, AI_STATIC,
  213, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_119_output_array, NULL)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  nl_128_nl_output, AI_STATIC,
  214, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_128_nl_output_array, NULL)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  nl_134_nl_output, AI_STATIC,
  215, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 40000),
  1, &nl_134_nl_output_array, NULL)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  nl_135_nl_output, AI_STATIC,
  216, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 32, 40000),
  1, &nl_135_nl_output_array, NULL)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  nl_138_output, AI_STATIC,
  217, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &nl_138_output_array, NULL)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  nl_144_nl_output, AI_STATIC,
  218, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 5), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &nl_144_nl_output_array, NULL)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  nl_150_output, AI_STATIC,
  219, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 5), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &nl_150_output_array, NULL)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  nl_21_nl_output, AI_STATIC,
  220, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &nl_21_nl_output_array, NULL)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  nl_22_nl_output, AI_STATIC,
  221, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &nl_22_nl_output_array, NULL)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  nl_24_nl_output, AI_STATIC,
  222, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &nl_24_nl_output_array, NULL)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  nl_26_nl_output, AI_STATIC,
  223, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &nl_26_nl_output_array, NULL)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  nl_28_nl_output, AI_STATIC,
  224, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &nl_28_nl_output_array, NULL)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  nl_32_nl_output, AI_STATIC,
  225, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &nl_32_nl_output_array, NULL)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  nl_33_nl_output, AI_STATIC,
  226, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &nl_33_nl_output_array, NULL)

/* Tensor #227 */
AI_TENSOR_OBJ_DECLARE(
  nl_35_nl_output, AI_STATIC,
  227, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &nl_35_nl_output_array, NULL)

/* Tensor #228 */
AI_TENSOR_OBJ_DECLARE(
  nl_37_nl_output, AI_STATIC,
  228, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 6000),
  1, &nl_37_nl_output_array, NULL)

/* Tensor #229 */
AI_TENSOR_OBJ_DECLARE(
  nl_3_nl_output, AI_STATIC,
  229, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 3), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &nl_3_nl_output_array, NULL)

/* Tensor #230 */
AI_TENSOR_OBJ_DECLARE(
  nl_41_nl_output, AI_STATIC,
  230, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 50, 1), AI_STRIDE_INIT(4, 4, 4, 32, 1600),
  1, &nl_41_nl_output_array, NULL)

/* Tensor #231 */
AI_TENSOR_OBJ_DECLARE(
  nl_42_nl_output, AI_STATIC,
  231, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 50, 1), AI_STRIDE_INIT(4, 4, 4, 32, 1600),
  1, &nl_42_nl_output_array, NULL)

/* Tensor #232 */
AI_TENSOR_OBJ_DECLARE(
  nl_42_nl_output0, AI_STATIC,
  232, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 50), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &nl_42_nl_output_array, NULL)

/* Tensor #233 */
AI_TENSOR_OBJ_DECLARE(
  nl_45_nl_output, AI_STATIC,
  233, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 10, 1), AI_STRIDE_INIT(4, 4, 4, 40, 400),
  1, &nl_45_nl_output_array, NULL)

/* Tensor #234 */
AI_TENSOR_OBJ_DECLARE(
  nl_46_nl_output, AI_STATIC,
  234, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 10, 1), AI_STRIDE_INIT(4, 4, 4, 40, 400),
  1, &nl_46_nl_output_array, NULL)

/* Tensor #235 */
AI_TENSOR_OBJ_DECLARE(
  nl_46_nl_output0, AI_STATIC,
  235, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 10), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_46_nl_output_array, NULL)

/* Tensor #236 */
AI_TENSOR_OBJ_DECLARE(
  nl_48_nl_output, AI_STATIC,
  236, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_48_nl_output_array, NULL)

/* Tensor #237 */
AI_TENSOR_OBJ_DECLARE(
  nl_4_nl_output, AI_STATIC,
  237, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 3), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &nl_4_nl_output_array, NULL)

/* Tensor #238 */
AI_TENSOR_OBJ_DECLARE(
  nl_54_output, AI_STATIC,
  238, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_54_output_array, NULL)

/* Tensor #239 */
AI_TENSOR_OBJ_DECLARE(
  nl_62_nl_output, AI_STATIC,
  239, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &nl_62_nl_output_array, NULL)

/* Tensor #240 */
AI_TENSOR_OBJ_DECLARE(
  nl_63_nl_output, AI_STATIC,
  240, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 50, 1), AI_STRIDE_INIT(4, 4, 4, 40, 2000),
  1, &nl_63_nl_output_array, NULL)

/* Tensor #241 */
AI_TENSOR_OBJ_DECLARE(
  nl_63_nl_output0, AI_STATIC,
  241, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 50), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_63_nl_output_array, NULL)

/* Tensor #242 */
AI_TENSOR_OBJ_DECLARE(
  nl_69_nl_output, AI_STATIC,
  242, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &nl_69_nl_output_array, NULL)

/* Tensor #243 */
AI_TENSOR_OBJ_DECLARE(
  nl_6_nl_output, AI_STATIC,
  243, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 50, 3), AI_STRIDE_INIT(4, 4, 4, 16, 800),
  1, &nl_6_nl_output_array, NULL)

/* Tensor #244 */
AI_TENSOR_OBJ_DECLARE(
  nl_70_nl_output, AI_STATIC,
  244, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &nl_70_nl_output_array, NULL)

/* Tensor #245 */
AI_TENSOR_OBJ_DECLARE(
  nl_73_output, AI_STATIC,
  245, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_73_output_array, NULL)

/* Tensor #246 */
AI_TENSOR_OBJ_DECLARE(
  nl_78_output, AI_STATIC,
  246, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_78_output_array, NULL)

/* Tensor #247 */
AI_TENSOR_OBJ_DECLARE(
  nl_86_nl_output, AI_STATIC,
  247, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 250, 1), AI_STRIDE_INIT(4, 4, 4, 40, 10000),
  1, &nl_86_nl_output_array, NULL)

/* Tensor #248 */
AI_TENSOR_OBJ_DECLARE(
  nl_86_nl_output0, AI_STATIC,
  248, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_86_nl_output_array, NULL)

/* Tensor #249 */
AI_TENSOR_OBJ_DECLARE(
  nl_86_nl_output1, AI_STATIC,
  249, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_86_nl_output_array, NULL)

/* Tensor #250 */
AI_TENSOR_OBJ_DECLARE(
  nl_95_nl_output, AI_STATIC,
  250, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &nl_95_nl_output_array, NULL)

/* Tensor #251 */
AI_TENSOR_OBJ_DECLARE(
  nl_96_nl_output, AI_STATIC,
  251, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 24, 30000),
  1, &nl_96_nl_output_array, NULL)

/* Tensor #252 */
AI_TENSOR_OBJ_DECLARE(
  nl_99_output, AI_STATIC,
  252, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &nl_99_output_array, NULL)

/* Tensor #253 */
AI_TENSOR_OBJ_DECLARE(
  nl_9_output, AI_STATIC,
  253, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 50, 1), AI_STRIDE_INIT(4, 4, 4, 12, 600),
  1, &nl_9_output_array, NULL)

/* Tensor #254 */
AI_TENSOR_OBJ_DECLARE(
  pool_103_output, AI_STATIC,
  254, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 4000),
  1, &pool_103_output_array, NULL)

/* Tensor #255 */
AI_TENSOR_OBJ_DECLARE(
  pool_103_output0, AI_STATIC,
  255, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 250), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &pool_103_output_array, NULL)

/* Tensor #256 */
AI_TENSOR_OBJ_DECLARE(
  pool_139_output, AI_STATIC,
  256, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 5, 1), AI_STRIDE_INIT(4, 4, 4, 16, 80),
  1, &pool_139_output_array, NULL)

/* Tensor #257 */
AI_TENSOR_OBJ_DECLARE(
  pool_139_output0, AI_STATIC,
  257, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 5), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &pool_139_output_array, NULL)

/* Tensor #258 */
AI_TENSOR_OBJ_DECLARE(
  pool_141_output, AI_STATIC,
  258, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 5, 1), AI_STRIDE_INIT(4, 4, 4, 16, 80),
  1, &pool_141_output_array, NULL)

/* Tensor #259 */
AI_TENSOR_OBJ_DECLARE(
  pool_141_output0, AI_STATIC,
  259, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 5), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &pool_141_output_array, NULL)

/* Tensor #260 */
AI_TENSOR_OBJ_DECLARE(
  pool_16_output, AI_STATIC,
  260, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &pool_16_output_array, NULL)

/* Tensor #261 */
AI_TENSOR_OBJ_DECLARE(
  pool_16_output0, AI_STATIC,
  261, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &pool_16_output_array, NULL)

/* Tensor #262 */
AI_TENSOR_OBJ_DECLARE(
  pool_17_output, AI_STATIC,
  262, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &pool_17_output_array, NULL)

/* Tensor #263 */
AI_TENSOR_OBJ_DECLARE(
  pool_17_output0, AI_STATIC,
  263, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &pool_17_output_array, NULL)

/* Tensor #264 */
AI_TENSOR_OBJ_DECLARE(
  pool_31_output, AI_STATIC,
  264, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 4000),
  1, &pool_31_output_array, NULL)

/* Tensor #265 */
AI_TENSOR_OBJ_DECLARE(
  pool_40_output, AI_STATIC,
  265, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 50, 1), AI_STRIDE_INIT(4, 4, 4, 24, 1200),
  1, &pool_40_output_array, NULL)

/* Tensor #266 */
AI_TENSOR_OBJ_DECLARE(
  pool_44_output, AI_STATIC,
  266, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 10, 1), AI_STRIDE_INIT(4, 4, 4, 32, 320),
  1, &pool_44_output_array, NULL)

/* Tensor #267 */
AI_TENSOR_OBJ_DECLARE(
  pool_5_output, AI_STATIC,
  267, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 50, 3), AI_STRIDE_INIT(4, 4, 4, 16, 800),
  1, &pool_5_output_array, NULL)

/* Tensor #268 */
AI_TENSOR_OBJ_DECLARE(
  pool_88_output, AI_STATIC,
  268, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 5, 1), AI_STRIDE_INIT(4, 4, 4, 40, 200),
  1, &pool_88_output_array, NULL)

/* Tensor #269 */
AI_TENSOR_OBJ_DECLARE(
  pool_88_output0, AI_STATIC,
  269, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 5), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &pool_88_output_array, NULL)

/* Tensor #270 */
AI_TENSOR_OBJ_DECLARE(
  reduce_15_output, AI_STATIC,
  270, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 16, 20000),
  1, &reduce_15_output_array, NULL)

/* Tensor #271 */
AI_TENSOR_OBJ_DECLARE(
  reduce_15_output0, AI_STATIC,
  271, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &reduce_15_output_array, NULL)

/* Tensor #272 */
AI_TENSOR_OBJ_DECLARE(
  split_52_num_or_size_splits, AI_STATIC,
  272, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &split_52_num_or_size_splits_array, NULL)

/* Tensor #273 */
AI_TENSOR_OBJ_DECLARE(
  split_52_output0, AI_STATIC,
  273, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &split_52_output0_array, NULL)

/* Tensor #274 */
AI_TENSOR_OBJ_DECLARE(
  split_52_output1, AI_STATIC,
  274, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &split_52_output1_array, NULL)

/* Tensor #275 */
AI_TENSOR_OBJ_DECLARE(
  tile_12_output, AI_STATIC,
  275, 0x0,
  AI_SHAPE_INIT(5, 1, 1, 50, 3, 25), AI_STRIDE_INIT(5, 4, 4, 100, 5000, 4),
  1, &tile_12_output_array, NULL)

/* Tensor #276 */
AI_TENSOR_OBJ_DECLARE(
  tile_12_output0, AI_STATIC,
  276, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1250, 3), AI_STRIDE_INIT(4, 4, 4, 4, 5000),
  1, &tile_12_output_array, NULL)

/* Tensor #277 */
AI_TENSOR_OBJ_DECLARE(
  tile_130_output, AI_STATIC,
  277, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 5, 250), AI_STRIDE_INIT(4, 4, 4, 40, 200),
  1, &tile_130_output_array, NULL)

/* Tensor #278 */
AI_TENSOR_OBJ_DECLARE(
  tile_130_output0, AI_STATIC,
  278, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &tile_130_output_array, NULL)

/* Tensor #279 */
AI_TENSOR_OBJ_DECLARE(
  tile_58_output, AI_STATIC,
  279, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 5, 10), AI_STRIDE_INIT(4, 4, 4, 88, 440),
  1, &tile_58_output_array, NULL)

/* Tensor #280 */
AI_TENSOR_OBJ_DECLARE(
  tile_58_output0, AI_STATIC,
  280, 0x0,
  AI_SHAPE_INIT(4, 1, 22, 1, 50), AI_STRIDE_INIT(4, 4, 4, 88, 88),
  1, &tile_58_output_array, NULL)

/* Tensor #281 */
AI_TENSOR_OBJ_DECLARE(
  tile_65_output, AI_STATIC,
  281, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 5, 50), AI_STRIDE_INIT(4, 4, 4, 40, 200),
  1, &tile_65_output_array, NULL)

/* Tensor #282 */
AI_TENSOR_OBJ_DECLARE(
  tile_65_output0, AI_STATIC,
  282, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &tile_65_output_array, NULL)

/* Tensor #283 */
AI_TENSOR_OBJ_DECLARE(
  tile_91_output, AI_STATIC,
  283, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 5, 250), AI_STRIDE_INIT(4, 4, 4, 40, 200),
  1, &tile_91_output_array, NULL)

/* Tensor #284 */
AI_TENSOR_OBJ_DECLARE(
  tile_91_output0, AI_STATIC,
  284, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &tile_91_output_array, NULL)

/* Tensor #285 */
AI_TENSOR_OBJ_DECLARE(
  transpose_10_output, AI_STATIC,
  285, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 50, 3), AI_STRIDE_INIT(4, 4, 4, 4, 200),
  1, &transpose_10_output_array, NULL)

/* Tensor #286 */
AI_TENSOR_OBJ_DECLARE(
  transpose_10_output0, AI_STATIC,
  286, 0x0,
  AI_SHAPE_INIT(5, 1, 1, 50, 3, 1), AI_STRIDE_INIT(5, 4, 4, 4, 200, 4),
  1, &transpose_10_output_array, NULL)

/* Tensor #287 */
AI_TENSOR_OBJ_DECLARE(
  transpose_148_output, AI_STATIC,
  287, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 5), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &transpose_148_output_array, NULL)

/* Tensor #288 */
AI_TENSOR_OBJ_DECLARE(
  transpose_8_output, AI_STATIC,
  288, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 50, 1), AI_STRIDE_INIT(4, 4, 4, 12, 600),
  1, &transpose_8_output_array, NULL)

/* Tensor #289 */
AI_TENSOR_OBJ_DECLARE(
  xs_0_Transpose_output, AI_STATIC,
  289, 0x0,
  AI_SHAPE_INIT(4, 1, 1250, 1, 3), AI_STRIDE_INIT(4, 4, 4, 5000, 5000),
  1, &xs_0_Transpose_output_array, NULL)

/* Tensor #290 */
AI_TENSOR_OBJ_DECLARE(
  xs_0_Transpose_output0, AI_STATIC,
  290, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1250, 3), AI_STRIDE_INIT(4, 4, 4, 4, 5000),
  1, &xs_0_Transpose_output_array, NULL)

/* Tensor #291 */
AI_TENSOR_OBJ_DECLARE(
  xs_0_output, AI_STATIC,
  291, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &xs_0_output_array, NULL)

/* Tensor #292 */
AI_TENSOR_OBJ_DECLARE(
  xs_1_output, AI_STATIC,
  292, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &xs_1_output_array, NULL)

/* Tensor #293 */
AI_TENSOR_OBJ_DECLARE(
  xs_2_output, AI_STATIC,
  293, 0x0,
  AI_SHAPE_INIT(4, 1, 216, 1, 1), AI_STRIDE_INIT(4, 4, 4, 864, 864),
  1, &xs_2_output_array, NULL)

/* Tensor #294 */
AI_TENSOR_OBJ_DECLARE(
  xs_3_output, AI_STATIC,
  294, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &xs_3_output_array, NULL)

/* Tensor #295 */
AI_TENSOR_OBJ_DECLARE(
  xs_4_output, AI_STATIC,
  295, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &xs_4_output_array, NULL)

/* Tensor #296 */
AI_TENSOR_OBJ_DECLARE(
  xs_5_output, AI_STATIC,
  296, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 250), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &xs_5_output_array, NULL)

/* Tensor #297 */
AI_TENSOR_OBJ_DECLARE(
  xs_6_output, AI_STATIC,
  297, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 50), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &xs_6_output_array, NULL)

/* Tensor #298 */
AI_TENSOR_OBJ_DECLARE(
  xs_7_output, AI_STATIC,
  298, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 10), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &xs_7_output_array, NULL)

/* Tensor #299 */
AI_TENSOR_OBJ_DECLARE(
  xs_8_output, AI_STATIC,
  299, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 250), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &xs_8_output_array, NULL)

/* Tensor #300 */
AI_TENSOR_OBJ_DECLARE(
  xs_9_output, AI_STATIC,
  300, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 5, 3), AI_STRIDE_INIT(4, 4, 4, 16, 80),
  1, &xs_9_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_150_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_149_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_150_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_150_layer, 150,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_150_chain,
  NULL, &nl_150_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_149_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_145_output, &transpose_148_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_149_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_149_layer, 149,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_149_chain,
  NULL, &nl_150_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_145_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_144_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_145_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_145_weights, &gemm_145_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_145_layer, 145,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_145_chain,
  NULL, &concat_149_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_144_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_144_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_144_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_144_nl_layer, 144,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_144_nl_chain,
  NULL, &gemm_145_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_144_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_143_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_144_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_144_weights, &gemm_144_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_144_layer, 144,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_144_chain,
  NULL, &nl_144_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_143_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &pool_141_output0, &pool_139_output0, &pool_88_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_143_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_143_layer, 143,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_143_chain,
  NULL, &gemm_144_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_88_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_86_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_88_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_88_layer, 88,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_88_chain,
  NULL, &concat_143_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(50, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(50, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_138_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_136_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_138_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_138_layer, 138,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_138_chain,
  NULL, &pool_88_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_136_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_135_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_136_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_136_weights, &conv2d_136_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_136_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_136_layer, 136,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_136_chain,
  NULL, &nl_138_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_135_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_135_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_135_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_135_nl_layer, 135,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_135_nl_chain,
  NULL, &conv2d_136_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_135_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_134_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_135_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_135_weights, &conv2d_135_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_135_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_135_layer, 135,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_135_chain,
  NULL, &nl_135_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_134_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_134_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_134_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_134_nl_layer, 134,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_134_nl_chain,
  NULL, &conv2d_135_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_134_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_132_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_134_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_134_weights, &conv2d_134_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_134_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_134_layer, 134,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_134_chain,
  NULL, &nl_134_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_132_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &tile_130_output0, &xs_4_output, &eltwise_29_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_132_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_132_layer, 132,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_132_chain,
  NULL, &conv2d_134_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_i16 tile_130_repeats_data[] = { 1, 5, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    tile_130_repeats, AI_ARRAY_FORMAT_S16,
    tile_130_repeats_data, tile_130_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  tile_130_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_128_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &tile_130_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  tile_130_layer, 130,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &tile_130_chain,
  NULL, &concat_132_layer, AI_STATIC, 
  .repeats = &tile_130_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_128_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_128_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_128_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_128_nl_layer, 128,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_128_nl_chain,
  NULL, &tile_130_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_128_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_110_output, &conv2d_126_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_128_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_128_layer, 128,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_128_chain,
  NULL, &nl_128_nl_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_126_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_124_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_126_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_126_weights, &conv2d_126_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_126_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_126_layer, 126,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_126_chain,
  NULL, &eltwise_128_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_124_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_122_output0, &eltwise_115_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_124_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_124_layer, 124,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_124_chain,
  NULL, &conv2d_126_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_115_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_112_output0, &nl_114_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_115_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_115_layer, 115,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_115_chain,
  NULL, &eltwise_124_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_114_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_112_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_114_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_114_layer, 114,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_114_chain,
  NULL, &eltwise_115_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_112_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_110_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_112_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_112_weights, &conv2d_75_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_112_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_112_layer, 112,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_112_chain,
  NULL, &nl_114_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_122_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_120_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_122_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_122_weights, &conv2d_75_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_122_layer, 122,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_122_chain,
  NULL, &conv2d_112_layer, AI_STATIC, 
  .groups = 10, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 127, 0, 127), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_120_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_117_output0, &nl_119_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_120_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_120_layer, 120,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_120_chain,
  NULL, &conv2d_122_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_119_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_117_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_119_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_119_layer, 119,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_119_chain,
  NULL, &eltwise_120_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_117_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_116_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_117_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_117_weights, &conv2d_117_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_117_layer, 117,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_117_chain,
  NULL, &nl_119_layer, AI_STATIC, 
  .groups = 10, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_116_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_110_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_116_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_116_weights, &conv2d_75_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_116_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_116_layer, 116,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_116_chain,
  NULL, &conv2d_117_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_110_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_109_output, &split_52_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_110_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_110_layer, 110,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_110_chain,
  NULL, &conv2d_116_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_109_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_107_nl_output0, &eltwise_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_109_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_109_layer, 109,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_109_chain,
  NULL, &eltwise_110_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_107_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_107_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_107_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_107_nl_layer, 107,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_107_nl_chain,
  NULL, &eltwise_109_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_107_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_105_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_107_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_107_weights, &conv2d_107_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_107_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_107_layer, 107,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_107_chain,
  NULL, &nl_107_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_105_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_86_nl_output1, &pool_103_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_105_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_105_layer, 105,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_105_chain,
  NULL, &conv2d_107_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_103_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_101_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_103_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_103_layer, 103,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_103_chain,
  NULL, &concat_105_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_101_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_55_output, &eltwise_100_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_101_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_101_layer, 101,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_101_chain,
  NULL, &pool_103_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_100_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_54_output, &nl_99_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_100_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_100_layer, 100,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_100_chain,
  NULL, &concat_101_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_99_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_97_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_99_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_99_layer, 99,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_99_chain,
  NULL, &eltwise_100_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_97_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_96_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_97_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_97_weights, &conv2d_97_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_97_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_97_layer, 97,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_97_chain,
  NULL, &nl_99_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_96_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_96_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_96_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_96_nl_layer, 96,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_96_nl_chain,
  NULL, &conv2d_97_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_96_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_95_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_96_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_96_weights, &conv2d_96_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_96_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_96_layer, 96,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_96_chain,
  NULL, &nl_96_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_95_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_95_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_95_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_95_nl_layer, 95,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_95_nl_chain,
  NULL, &conv2d_96_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_95_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_93_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_95_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_95_weights, &conv2d_95_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_95_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_95_layer, 95,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_95_chain,
  NULL, &nl_95_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_93_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &tile_91_output0, &xs_4_output, &eltwise_29_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_93_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_93_layer, 93,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_93_chain,
  NULL, &conv2d_95_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_i16 tile_91_repeats_data[] = { 1, 5, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    tile_91_repeats, AI_ARRAY_FORMAT_S16,
    tile_91_repeats_data, tile_91_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  tile_91_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_86_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &tile_91_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  tile_91_layer, 91,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &tile_91_chain,
  NULL, &concat_93_layer, AI_STATIC, 
  .repeats = &tile_91_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_86_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_86_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_86_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_86_nl_layer, 86,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_86_nl_chain,
  NULL, &tile_91_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_86_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_70_nl_output, &conv2d_85_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_86_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_86_layer, 86,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_86_chain,
  NULL, &nl_86_nl_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_85_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_83_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_85_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_85_weights, &conv2d_85_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_85_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_85_layer, 85,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_85_chain,
  NULL, &eltwise_86_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_81_output0, &eltwise_74_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_83_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_83_layer, 83,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_83_chain,
  NULL, &conv2d_85_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_74_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_71_output0, &nl_73_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_74_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_74_layer, 74,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_74_chain,
  NULL, &eltwise_83_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_73_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_71_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_73_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_73_layer, 73,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_73_chain,
  NULL, &eltwise_74_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_71_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_70_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_71_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_71_weights, &conv2d_75_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_71_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_71_layer, 71,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_71_chain,
  NULL, &nl_73_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_81_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_79_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_81_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_81_weights, &conv2d_75_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_81_layer, 81,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_81_chain,
  NULL, &conv2d_71_layer, AI_STATIC, 
  .groups = 10, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 127, 0, 127), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_79_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_76_output0, &nl_78_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_79_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_79_layer, 79,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_79_chain,
  NULL, &conv2d_81_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_78_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_76_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_78_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_78_layer, 78,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_78_chain,
  NULL, &eltwise_79_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_76_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_76_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_76_weights, &conv2d_76_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_76_layer, 76,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_76_chain,
  NULL, &nl_78_layer, AI_STATIC, 
  .groups = 10, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_75_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_70_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_75_weights, &conv2d_75_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_75_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_75_layer, 75,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_75_chain,
  NULL, &conv2d_76_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_70_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_70_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_70_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_70_nl_layer, 70,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_70_nl_chain,
  NULL, &conv2d_75_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_70_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_69_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_70_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_70_weights, &conv2d_70_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_70_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_70_layer, 70,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_70_chain,
  NULL, &nl_70_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_69_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_69_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_69_nl_layer, 69,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_69_nl_chain,
  NULL, &conv2d_70_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_69_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_67_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_69_weights, &conv2d_69_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_69_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_69_layer, 69,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_69_chain,
  NULL, &nl_69_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_67_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 4, &tile_65_output0, &xs_5_output, &eltwise_38_output0, &xs_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_67_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_67_layer, 67,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_67_chain,
  NULL, &conv2d_69_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_i16 tile_65_repeats_data[] = { 1, 5, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    tile_65_repeats, AI_ARRAY_FORMAT_S16,
    tile_65_repeats_data, tile_65_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  tile_65_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_63_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &tile_65_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  tile_65_layer, 65,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &tile_65_chain,
  NULL, &concat_67_layer, AI_STATIC, 
  .repeats = &tile_65_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_63_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_63_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_63_nl_layer, 63,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_63_nl_chain,
  NULL, &tile_65_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_63_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_62_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_63_weights, &conv2d_63_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_63_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_63_layer, 63,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_63_chain,
  NULL, &nl_63_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_62_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_62_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_62_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_62_nl_layer, 62,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_62_nl_chain,
  NULL, &conv2d_63_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_62_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_60_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_62_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_62_weights, &conv2d_62_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_62_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_62_layer, 62,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_62_chain,
  NULL, &nl_62_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_60_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &tile_58_output0, &xs_6_output, &nl_42_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_60_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_60_layer, 60,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_60_chain,
  NULL, &conv2d_62_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_i16 tile_58_repeats_data[] = { 1, 5, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    tile_58_repeats, AI_ARRAY_FORMAT_S16,
    tile_58_repeats_data, tile_58_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  tile_58_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &tile_58_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  tile_58_layer, 58,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &tile_58_chain,
  NULL, &concat_60_layer, AI_STATIC, 
  .repeats = &tile_58_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_56_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &xs_7_output, &nl_46_nl_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_56_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_56_layer, 56,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_56_chain,
  NULL, &tile_58_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_46_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_46_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_46_nl_layer, 46,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_46_nl_chain,
  NULL, &concat_56_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_45_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_46_weights, &conv2d_46_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_46_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_46_layer, 46,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_46_chain,
  NULL, &nl_46_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_45_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_45_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_45_nl_layer, 45,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_45_nl_chain,
  NULL, &conv2d_46_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_44_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_45_weights, &conv2d_45_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_45_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_45_layer, 45,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_45_chain,
  NULL, &nl_45_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_44_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_42_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_44_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_44_layer, 44,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_44_chain,
  NULL, &conv2d_45_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
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
  NULL, &pool_44_layer, AI_STATIC, 
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
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
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
  .filter_pad = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_40_layer, 40,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_40_chain,
  NULL, &conv2d_41_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_36_output, &nl_37_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_38_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_38_layer, 38,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_38_chain,
  NULL, &pool_40_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
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
  NULL, &eltwise_38_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_37_weights, &conv2d_37_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_37_layer, 37,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &conv2d_37_chain,
  NULL, &nl_37_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(4, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 8, 0, 8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_34_output, &nl_35_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_36_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_36_layer, 36,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_36_chain,
  NULL, &conv2d_37_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_35_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_35_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_35_nl_layer, 35,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_35_nl_chain,
  NULL, &eltwise_36_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_35_weights, &conv2d_35_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_35_layer, 35,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &conv2d_35_chain,
  NULL, &nl_35_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(2, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 4, 0, 4), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_32_nl_output, &nl_33_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_34_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_34_layer, 34,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_34_chain,
  NULL, &conv2d_35_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
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
  NULL, &eltwise_34_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_32_nl_output),
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
  nl_32_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_32_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_32_nl_layer, 32,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_32_nl_chain,
  NULL, &conv2d_33_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_32_weights, &conv2d_32_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_32_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_32_layer, 32,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_32_chain,
  NULL, &nl_32_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_31_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_31_layer, 31,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_31_chain,
  NULL, &conv2d_32_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_27_output, &nl_28_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_29_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_29_layer, 29,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_29_chain,
  NULL, &pool_31_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_28_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_28_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_28_nl_layer, 28,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_28_nl_chain,
  NULL, &eltwise_29_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_28_weights, &conv2d_28_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_28_layer, 28,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &conv2d_28_chain,
  NULL, &nl_28_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(8, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 24, 0, 24), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_25_output, &nl_26_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_27_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_27_layer, 27,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_27_chain,
  NULL, &conv2d_28_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_26_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_26_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_26_nl_layer, 26,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_26_nl_chain,
  NULL, &eltwise_27_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_26_weights, &conv2d_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_26_layer, 26,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &conv2d_26_chain,
  NULL, &nl_26_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(4, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 12, 0, 12), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_23_output, &nl_24_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_25_layer, 25,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_25_chain,
  NULL, &conv2d_26_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_24_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_24_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_24_nl_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_24_nl_chain,
  NULL, &eltwise_25_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_24_weights, &conv2d_24_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_24_layer, 24,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &conv2d_24_chain,
  NULL, &nl_24_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(2, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 6, 0, 6), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_21_nl_output, &nl_22_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_23_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_23_layer, 23,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_23_chain,
  NULL, &conv2d_24_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_22_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_22_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_22_nl_layer, 22,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_22_nl_chain,
  NULL, &eltwise_23_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_21_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_22_weights, &conv2d_22_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_22_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_22_layer, 22,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_22_chain,
  NULL, &nl_22_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_21_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_21_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_21_nl_layer, 21,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_21_nl_chain,
  NULL, &conv2d_22_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_19_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_21_weights, &conv2d_21_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_21_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_21_layer, 21,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_21_chain,
  NULL, &nl_21_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_18_output, &xs_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_19_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_19_layer, 19,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_19_chain,
  NULL, &conv2d_21_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &reduce_15_output0, &pool_16_output0, &pool_17_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_18_layer, 18,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_18_chain,
  NULL, &concat_19_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_float reduce_15_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_15_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_15_neutral_value_data, reduce_15_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_15_layer, 15,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_15_chain,
  NULL, &concat_18_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_15_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tile_12_output0, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_14_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_14_layer, 14,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_14_chain,
  NULL, &reduce_15_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)


AI_STATIC_CONST ai_i16 tile_12_repeats_data[] = { 1, 1, 1, 25, 1 };
AI_ARRAY_OBJ_DECLARE(
    tile_12_repeats, AI_ARRAY_FORMAT_S16,
    tile_12_repeats_data, tile_12_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  tile_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_10_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &tile_12_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  tile_12_layer, 12,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &tile_12_chain,
  NULL, &eltwise_14_layer, AI_STATIC, 
  .repeats = &tile_12_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_10_layer, 10,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_10_chain,
  NULL, &tile_12_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_9_layer, 9,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_9_chain,
  NULL, &transpose_10_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_8_layer, 8,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_8_chain,
  NULL, &nl_9_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_6_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_7_layer, 7,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_7_chain,
  NULL, &transpose_8_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_6_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_6_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_6_nl_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_6_nl_chain,
  NULL, &gemm_7_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_6_layer, 6,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_6_chain,
  NULL, &nl_6_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_5_layer, 5,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_5_chain,
  NULL, &gemm_6_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(25, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(25, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_16_layer, 16,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_16_chain,
  NULL, &pool_5_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 3), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_17_layer, 17,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_17_chain,
  NULL, &pool_16_layer, AI_STATIC, 
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
  NULL, &pool_17_layer, AI_STATIC, 
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
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
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
  .filter_pad = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
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

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_52_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_53_scale, &eltwise_53_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_53_layer, 53,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &eltwise_53_chain,
  NULL, &xs_0_Transpose_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  split_52_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_52_output0, &split_52_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_52_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  split_52_layer, 52,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &split_52_chain,
  NULL, &eltwise_53_layer, AI_STATIC, 
  .outer_elems = 1, 
  .outer_elems_stride = 80, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_48_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_49_weights, &gemm_49_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_49_layer, 49,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_49_chain,
  NULL, &split_52_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_48_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_48_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_48_nl_layer, 48,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_48_nl_chain,
  NULL, &gemm_49_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &xs_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_48_weights, &gemm_48_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_48_layer, 48,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_48_chain,
  NULL, &nl_48_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &arith_constant116_3D, &nl_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_55_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_55_layer, 55,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_55_chain,
  NULL, &gemm_48_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &xs_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_54_layer, 54,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_54_chain,
  NULL, &eltwise_55_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_139_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &xs_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_139_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_139_layer, 139,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_139_chain,
  NULL, &nl_54_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 3), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_141_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &xs_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_141_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_141_layer, 141,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_141_chain,
  NULL, &pool_139_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 3), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_148_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_146_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_148_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_148_layer, 148,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_148_chain,
  NULL, &pool_141_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_146_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &xs_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_146_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_146_weights, &gemm_146_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_146_layer, 146,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_146_chain,
  NULL, &transpose_148_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 73476, 1, 1),
    73476, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 266784, 1, 1),
    266784, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECGB_IN_NUM, &xs_0_output, &xs_1_output, &xs_2_output, &xs_3_output, &xs_7_output, &xs_6_output, &xs_5_output, &xs_8_output, &xs_4_output, &xs_9_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECGB_OUT_NUM, &nl_138_output, &concat_101_output0, &nl_150_output),
  &gemm_146_layer, 0xd1910620, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 73476, 1, 1),
      73476, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 266784, 1, 1),
      266784, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECGB_IN_NUM, &xs_0_output, &xs_1_output, &xs_2_output, &xs_3_output, &xs_7_output, &xs_6_output, &xs_5_output, &xs_8_output, &xs_4_output, &xs_9_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_ECGB_OUT_NUM, &nl_138_output, &concat_101_output0, &nl_150_output),
  &gemm_146_layer, 0xd1910620, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool ecgb_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_ecgb_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    xs_0_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    xs_0_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    xs_1_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 15000);
    xs_1_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 15000);
    xs_2_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    xs_2_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    xs_3_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20864);
    xs_3_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20864);
    xs_7_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    xs_7_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    xs_6_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 26344);
    xs_6_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 26344);
    xs_5_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 28344);
    xs_5_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 28344);
    xs_8_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 36344);
    xs_8_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 36344);
    xs_4_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 46344);
    xs_4_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 46344);
    xs_9_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76344);
    xs_9_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76344);
    gemm_146_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76584);
    gemm_146_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76584);
    transpose_148_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76644);
    transpose_148_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76644);
    pool_141_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76704);
    pool_141_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76704);
    pool_139_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76344);
    pool_139_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76344);
    nl_54_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20864);
    nl_54_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20864);
    eltwise_55_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76784);
    eltwise_55_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76784);
    gemm_48_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76424);
    gemm_48_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76424);
    nl_48_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    nl_48_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    gemm_49_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20040);
    gemm_49_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20040);
    split_52_output0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    split_52_output0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    split_52_output1_array.data = AI_PTR(g_ecgb_activations_map[0] + 20120);
    split_52_output1_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20120);
    eltwise_53_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20040);
    eltwise_53_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20040);
    xs_0_Transpose_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    xs_0_Transpose_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_3_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_3_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_3_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 96784);
    conv2d_3_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 96784);
    nl_3_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 96784);
    nl_3_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 96784);
    conv2d_4_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_4_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 156784);
    conv2d_4_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 156784);
    nl_4_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_4_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    pool_17_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 141784);
    pool_17_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 141784);
    pool_16_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 161784);
    pool_16_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 161784);
    pool_5_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    pool_5_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    gemm_6_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 2400);
    gemm_6_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 2400);
    nl_6_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    nl_6_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    gemm_7_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 2400);
    gemm_7_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 2400);
    transpose_8_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    transpose_8_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    nl_9_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 600);
    nl_9_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 600);
    transpose_10_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    transpose_10_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    tile_12_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 181784);
    tile_12_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 181784);
    eltwise_14_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    eltwise_14_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    reduce_15_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 181784);
    reduce_15_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 181784);
    concat_18_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    concat_18_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    concat_19_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 141784);
    concat_19_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 141784);
    conv2d_21_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_21_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_21_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_21_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_21_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    nl_21_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_22_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_22_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_22_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_22_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_22_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 101784);
    nl_22_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 101784);
    eltwise_23_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    eltwise_23_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_24_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_24_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    nl_24_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 101784);
    nl_24_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 101784);
    eltwise_25_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    eltwise_25_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_26_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_26_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_26_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 101784);
    nl_26_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 101784);
    eltwise_27_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    eltwise_27_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_28_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_28_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    nl_28_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 101784);
    nl_28_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 101784);
    eltwise_29_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    eltwise_29_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    pool_31_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    pool_31_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_32_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_32_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_32_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 85784);
    conv2d_32_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 85784);
    nl_32_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 91784);
    nl_32_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 91784);
    conv2d_33_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_33_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_33_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_33_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_33_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 97784);
    nl_33_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 97784);
    eltwise_34_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    eltwise_34_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_35_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    conv2d_35_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    nl_35_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 93784);
    nl_35_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 93784);
    eltwise_36_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    eltwise_36_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    conv2d_37_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_37_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_37_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 93784);
    nl_37_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 93784);
    eltwise_38_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    eltwise_38_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    pool_40_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    pool_40_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    conv2d_41_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_41_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_41_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 88984);
    conv2d_41_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 88984);
    nl_41_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 90584);
    nl_41_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 90584);
    conv2d_42_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_42_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_42_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    conv2d_42_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    nl_42_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 89384);
    nl_42_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 89384);
    pool_44_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    pool_44_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_45_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20480);
    conv2d_45_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20480);
    conv2d_45_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    conv2d_45_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    nl_45_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    nl_45_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_46_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20560);
    conv2d_46_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20560);
    conv2d_46_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    conv2d_46_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    nl_46_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    nl_46_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    concat_56_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    concat_56_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    tile_58_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 90984);
    tile_58_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 90984);
    concat_60_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 95384);
    concat_60_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 95384);
    conv2d_62_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_62_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_62_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    conv2d_62_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    nl_62_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    nl_62_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_63_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_63_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_63_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    conv2d_63_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    nl_63_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    nl_63_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    tile_65_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 87784);
    tile_65_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 87784);
    concat_67_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 97784);
    concat_67_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 97784);
    conv2d_69_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_69_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_69_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_69_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    nl_69_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    nl_69_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_70_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_70_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_70_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_70_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    nl_70_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    nl_70_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_75_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_75_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_75_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_75_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_76_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_76_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_78_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    nl_78_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    eltwise_79_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 91784);
    eltwise_79_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 91784);
    conv2d_81_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_81_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_71_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_71_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_71_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_71_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_73_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 91784);
    nl_73_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 91784);
    eltwise_74_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 101784);
    eltwise_74_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 101784);
    eltwise_83_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    eltwise_83_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_85_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_85_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_85_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    conv2d_85_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    eltwise_86_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    eltwise_86_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_86_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 25864);
    nl_86_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 25864);
    tile_91_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    tile_91_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    concat_93_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 131784);
    concat_93_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 131784);
    conv2d_95_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_95_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_95_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_95_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_95_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 111784);
    nl_95_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 111784);
    conv2d_96_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_96_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_96_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_96_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_96_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 111784);
    nl_96_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 111784);
    conv2d_97_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_97_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_97_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    conv2d_97_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    nl_99_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 96784);
    nl_99_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 96784);
    eltwise_100_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 81784);
    eltwise_100_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 81784);
    concat_101_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 96784);
    concat_101_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 96784);
    pool_103_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    pool_103_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    concat_105_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76784);
    concat_105_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76784);
    conv2d_107_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_107_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20160);
    conv2d_107_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_107_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    nl_107_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76784);
    nl_107_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76784);
    eltwise_109_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    eltwise_109_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    eltwise_110_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 76784);
    eltwise_110_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 76784);
    conv2d_116_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_116_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_116_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_116_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_117_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 86784);
    conv2d_117_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 86784);
    nl_119_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    nl_119_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    eltwise_120_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 116784);
    eltwise_120_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 116784);
    conv2d_122_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_122_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_112_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_112_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_112_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 86784);
    conv2d_112_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 86784);
    nl_114_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 116784);
    nl_114_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 116784);
    eltwise_115_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 126784);
    eltwise_115_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 126784);
    eltwise_124_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 86784);
    eltwise_124_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 86784);
    conv2d_126_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_126_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20000);
    conv2d_126_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_126_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    eltwise_128_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 86784);
    eltwise_128_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 86784);
    nl_128_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    nl_128_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    tile_130_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 116784);
    tile_130_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 116784);
    concat_132_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 166784);
    concat_132_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 166784);
    conv2d_134_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_134_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_134_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_134_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    nl_134_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 116784);
    nl_134_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 116784);
    conv2d_135_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_135_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_135_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    conv2d_135_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    nl_135_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 116784);
    nl_135_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 116784);
    conv2d_136_scratch0_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_136_scratch0_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    conv2d_136_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 32);
    conv2d_136_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 32);
    nl_138_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 35864);
    nl_138_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 35864);
    pool_88_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    pool_88_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    concat_143_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 200);
    concat_143_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 200);
    gemm_144_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    gemm_144_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    nl_144_nl_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 80);
    nl_144_nl_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 80);
    gemm_145_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 0);
    gemm_145_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 0);
    concat_149_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 20);
    concat_149_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 20);
    nl_150_output_array.data = AI_PTR(g_ecgb_activations_map[0] + 100);
    nl_150_output_array.data_start = AI_PTR(g_ecgb_activations_map[0] + 100);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool ecgb_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_ecgb_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    arith_constant116_3D_array.format |= AI_FMT_FLAG_CONST;
    arith_constant116_3D_array.data = AI_PTR(g_ecgb_weights_map[0] + 0);
    arith_constant116_3D_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 0);
    gemm_146_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_146_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 4);
    gemm_146_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 4);
    gemm_146_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_146_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 20);
    gemm_146_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 20);
    gemm_48_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_48_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 24);
    gemm_48_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 24);
    gemm_48_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_48_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 8664);
    gemm_48_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 8664);
    gemm_49_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_49_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 8704);
    gemm_49_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 8704);
    gemm_49_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_49_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 9504);
    gemm_49_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 9504);
    split_52_num_or_size_splits_array.format |= AI_FMT_FLAG_CONST;
    split_52_num_or_size_splits_array.data = AI_PTR(g_ecgb_weights_map[0] + 9584);
    split_52_num_or_size_splits_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 9584);
    eltwise_53_scale_array.format |= AI_FMT_FLAG_CONST;
    eltwise_53_scale_array.data = AI_PTR(g_ecgb_weights_map[0] + 9588);
    eltwise_53_scale_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 9588);
    eltwise_53_bias_array.format |= AI_FMT_FLAG_CONST;
    eltwise_53_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 9628);
    eltwise_53_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 9628);
    conv2d_3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 9668);
    conv2d_3_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 9668);
    conv2d_3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 9780);
    conv2d_3_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 9780);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 9796);
    conv2d_4_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 9796);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 10244);
    conv2d_4_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 10244);
    gemm_6_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_6_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 10260);
    gemm_6_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 10260);
    gemm_6_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_6_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 10324);
    gemm_6_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 10324);
    gemm_7_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_7_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 10340);
    gemm_7_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 10340);
    gemm_7_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_7_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 10356);
    gemm_7_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 10356);
    conv2d_21_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 10360);
    conv2d_21_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 10360);
    conv2d_21_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 10568);
    conv2d_21_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 10568);
    conv2d_22_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 10584);
    conv2d_22_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 10584);
    conv2d_22_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 11032);
    conv2d_22_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 11032);
    conv2d_24_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_24_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 11048);
    conv2d_24_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 11048);
    conv2d_24_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_24_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 11496);
    conv2d_24_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 11496);
    conv2d_26_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 11512);
    conv2d_26_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 11512);
    conv2d_26_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 11960);
    conv2d_26_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 11960);
    conv2d_28_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_28_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 11976);
    conv2d_28_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 11976);
    conv2d_28_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_28_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 12424);
    conv2d_28_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 12424);
    conv2d_32_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_32_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 12440);
    conv2d_32_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 12440);
    conv2d_32_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_32_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 12536);
    conv2d_32_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 12536);
    conv2d_33_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_33_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 12560);
    conv2d_33_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 12560);
    conv2d_33_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_33_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 13280);
    conv2d_33_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 13280);
    conv2d_35_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_35_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 13304);
    conv2d_35_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 13304);
    conv2d_35_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_35_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 14024);
    conv2d_35_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 14024);
    conv2d_37_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_37_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 14048);
    conv2d_37_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 14048);
    conv2d_37_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_37_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 14768);
    conv2d_37_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 14768);
    conv2d_41_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_41_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 14792);
    conv2d_41_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 14792);
    conv2d_41_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_41_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 15752);
    conv2d_41_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 15752);
    conv2d_42_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_42_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 15784);
    conv2d_42_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 15784);
    conv2d_42_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_42_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 17064);
    conv2d_42_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 17064);
    conv2d_45_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_45_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 17096);
    conv2d_45_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 17096);
    conv2d_45_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_45_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 18056);
    conv2d_45_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 18056);
    conv2d_46_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_46_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 18096);
    conv2d_46_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 18096);
    conv2d_46_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_46_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 19296);
    conv2d_46_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 19296);
    conv2d_62_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_62_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 19336);
    conv2d_62_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 19336);
    conv2d_62_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_62_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 27336);
    conv2d_62_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 27336);
    conv2d_63_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_63_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 27376);
    conv2d_63_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 27376);
    conv2d_63_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_63_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 29376);
    conv2d_63_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 29376);
    conv2d_69_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_69_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 29416);
    conv2d_69_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 29416);
    conv2d_69_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_69_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 36216);
    conv2d_69_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 36216);
    conv2d_70_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_70_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 36256);
    conv2d_70_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 36256);
    conv2d_70_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_70_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 38256);
    conv2d_70_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 38256);
    conv2d_75_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_75_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 38296);
    conv2d_75_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 38296);
    conv2d_75_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_75_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 38696);
    conv2d_75_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 38696);
    conv2d_76_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_76_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 38736);
    conv2d_76_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 38736);
    conv2d_76_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_76_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 38856);
    conv2d_76_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 38856);
    conv2d_81_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_81_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 38896);
    conv2d_81_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 38896);
    conv2d_71_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_71_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 49096);
    conv2d_71_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 49096);
    conv2d_85_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_85_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 49496);
    conv2d_85_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 49496);
    conv2d_85_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_85_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 49896);
    conv2d_85_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 49896);
    conv2d_95_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_95_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 49936);
    conv2d_95_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 49936);
    conv2d_95_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_95_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 53296);
    conv2d_95_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 53296);
    conv2d_96_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_96_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 53320);
    conv2d_96_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 53320);
    conv2d_96_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_96_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 54328);
    conv2d_96_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 54328);
    conv2d_97_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_97_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 54352);
    conv2d_97_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 54352);
    conv2d_97_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_97_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 54424);
    conv2d_97_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 54424);
    conv2d_107_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_107_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 54436);
    conv2d_107_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 54436);
    conv2d_107_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_107_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 54996);
    conv2d_107_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 54996);
    conv2d_116_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_116_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 55036);
    conv2d_116_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 55036);
    conv2d_117_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_117_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 55436);
    conv2d_117_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 55436);
    conv2d_117_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_117_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 55556);
    conv2d_117_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 55556);
    conv2d_122_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_122_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 55596);
    conv2d_122_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 55596);
    conv2d_112_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_112_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 65796);
    conv2d_112_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 65796);
    conv2d_126_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_126_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 66196);
    conv2d_126_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 66196);
    conv2d_126_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_126_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 66596);
    conv2d_126_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 66596);
    conv2d_134_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_134_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 66636);
    conv2d_134_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 66636);
    conv2d_134_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_134_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 71116);
    conv2d_134_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 71116);
    conv2d_135_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_135_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 71148);
    conv2d_135_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 71148);
    conv2d_135_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_135_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 72940);
    conv2d_135_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 72940);
    conv2d_136_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_136_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 72972);
    conv2d_136_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 72972);
    conv2d_136_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_136_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 73132);
    conv2d_136_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 73132);
    gemm_144_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_144_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 73152);
    gemm_144_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 73152);
    gemm_144_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_144_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 73440);
    gemm_144_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 73440);
    gemm_145_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_145_weights_array.data = AI_PTR(g_ecgb_weights_map[0] + 73456);
    gemm_145_weights_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 73456);
    gemm_145_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_145_bias_array.data = AI_PTR(g_ecgb_weights_map[0] + 73472);
    gemm_145_bias_array.data_start = AI_PTR(g_ecgb_weights_map[0] + 73472);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_ecgb_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_ECGB_MODEL_NAME,
      .model_signature   = AI_ECGB_MODEL_SIGNATURE,
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
      
      .n_macc            = 7386047,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xd1910620,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_ecgb_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_ECGB_MODEL_NAME,
      .model_signature   = AI_ECGB_MODEL_SIGNATURE,
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
      
      .n_macc            = 7386047,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xd1910620,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_ecgb_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_ecgb_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_ecgb_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_ecgb_create(network, AI_ECGB_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_ecgb_data_params_get(&params) != true) {
    err = ai_ecgb_get_error(*network);
    return err;
  }
#if defined(AI_ECGB_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_ECGB_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_ecgb_init(*network, &params) != true) {
    err = ai_ecgb_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_ecgb_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_ecgb_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_ecgb_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_ecgb_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= ecgb_configure_weights(net_ctx, params);
  ok &= ecgb_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_ecgb_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_ecgb_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_ECGB_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

