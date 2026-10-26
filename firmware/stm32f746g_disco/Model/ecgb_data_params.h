/**
  ******************************************************************************
  * @file    ecgb_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-25T22:15:55+0700
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

#ifndef ECGB_DATA_PARAMS_H
#define ECGB_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_ECGB_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_ecgb_data_weights_params[1]))
*/

#define AI_ECGB_DATA_CONFIG               (NULL)


#define AI_ECGB_DATA_ACTIVATIONS_SIZES \
  { 266784, }
#define AI_ECGB_DATA_ACTIVATIONS_SIZE     (266784)
#define AI_ECGB_DATA_ACTIVATIONS_COUNT    (1)
#define AI_ECGB_DATA_ACTIVATION_1_SIZE    (266784)



#define AI_ECGB_DATA_WEIGHTS_SIZES \
  { 73476, }
#define AI_ECGB_DATA_WEIGHTS_SIZE         (73476)
#define AI_ECGB_DATA_WEIGHTS_COUNT        (1)
#define AI_ECGB_DATA_WEIGHT_1_SIZE        (73476)



#define AI_ECGB_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_ecgb_activations_table[1])

extern ai_handle g_ecgb_activations_table[1 + 2];



#define AI_ECGB_DATA_WEIGHTS_TABLE_GET() \
  (&g_ecgb_weights_table[1])

extern ai_handle g_ecgb_weights_table[1 + 2];


#endif    /* ECGB_DATA_PARAMS_H */
