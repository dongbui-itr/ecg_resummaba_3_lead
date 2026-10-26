/* Written by export_model.py - do not edit. */
#ifndef MODEL_META_H
#define MODEL_META_H

#define ECG_MODEL_TAG        "r30k_A_e36"
#define ECG_MODEL_SHA256     "9e4865dbfdb26692682a393821751981c13b8284fc54e769334b7301ea0d1ff9"
#define ECG_MODEL_MACC       13584068u
#define ECG_MODEL_WEIGHTS    115860u
#define ECG_MODEL_ACT_A      360000u
#define ECG_MODEL_ACT_B      266784u

/* QRST cancellation / RR autocorrelation stage (App/ecg_stage.c) */
#define STAGE_PRE            12      /* template steps before R */
#define STAGE_POST           56     /* ... and after */
#define STAGE_NMS            51      /* beat NMS window, steps */
#define STAGE_MIN_PROB       0.300000f
#define STAGE_MASK           17     /* QRS mask window, steps */
#define STAGE_RR_LO          34    /* first lag */
#define STAGE_RR_HI          250    /* one past the last lag */

/* net A outputs, in C order (= TFLite order) */
#define ECGA_OUT_POS_LOGIT  0
#define ECGA_OUT_P_POS      1
#define ECGA_OUT_V1         2
#define ECGA_OUT_V2         3
#define ECGA_OUT_V3         4
#define ECGA_OUT_VB         5
#define ECGA_OUT_VS         6
#define ECGA_OUT_CHAN       7

/* net B inputs by meaning */
#define ECGB_IN_RESIDUAL   0
#define ECGB_IN_QRS        1
#define ECGB_IN_RR         2
#define ECGB_IN_POS_LOGIT  3
#define ECGB_IN_V1         4
#define ECGB_IN_V2         5
#define ECGB_IN_V3         6
#define ECGB_IN_VB         7
#define ECGB_IN_VS         8
#define ECGB_IN_CHAN       9
/* C input slot -> ECGB_IN_* it takes (X-CUBE-AI orders the inputs itself) */
#define ECGB_C_INPUT_ORDER   { ECGB_IN_RESIDUAL, ECGB_IN_QRS, ECGB_IN_RR, ECGB_IN_POS_LOGIT, ECGB_IN_VB, ECGB_IN_V3, ECGB_IN_V2, ECGB_IN_VS, ECGB_IN_V1, ECGB_IN_CHAN }

#endif
