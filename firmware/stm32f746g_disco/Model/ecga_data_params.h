/**
  ******************************************************************************
  * @file    ecga_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-25T22:15:20+0700
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef ECGA_DATA_PARAMS_H
#define ECGA_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_ECGA_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_ecga_data_weights_params[1]))
*/

#define AI_ECGA_DATA_CONFIG               (NULL)


#define AI_ECGA_DATA_ACTIVATIONS_SIZES \
  { 360000, }
#define AI_ECGA_DATA_ACTIVATIONS_SIZE     (360000)
#define AI_ECGA_DATA_ACTIVATIONS_COUNT    (1)
#define AI_ECGA_DATA_ACTIVATION_1_SIZE    (360000)



#define AI_ECGA_DATA_WEIGHTS_SIZES \
  { 42384, }
#define AI_ECGA_DATA_WEIGHTS_SIZE         (42384)
#define AI_ECGA_DATA_WEIGHTS_COUNT        (1)
#define AI_ECGA_DATA_WEIGHT_1_SIZE        (42384)



#define AI_ECGA_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_ecga_activations_table[1])

extern ai_handle g_ecga_activations_table[1 + 2];



#define AI_ECGA_DATA_WEIGHTS_TABLE_GET() \
  (&g_ecga_weights_table[1])

extern ai_handle g_ecga_weights_table[1 + 2];


#endif    /* ECGA_DATA_PARAMS_H */
