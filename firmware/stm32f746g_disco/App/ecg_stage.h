/*
 * The part of the dual U-Net that runs as plain C between net A and net B (X-CUBE-AI
 * miscompiles it): QRST cancellation (dualunet.QRSTCancel) and the RR autocorrelation
 * (layers.RhythmDescriptor). Reference: ecgr/rhythm/mcu_export.py c_stage(); constants in
 * Model/model_meta.h are read from the checkpoint's own layers.
 */
#ifndef ECG_STAGE_H
#define ECG_STAGE_H

#include "model_meta.h"

#define STAGE_IN_SAMPLES 2500
#define STAGE_LEADS      3
#define STAGE_STEPS      1250                 /* 125 Hz, 2 input samples per step */
#define STAGE_N_LAGS     (STAGE_RR_HI - STAGE_RR_LO)

/* x: (2500, 3) model input, p: (1250,) p(beat) from net A.
 * residual: (1250, 3) signal minus the average beat; qrs: (1250,) 1 near a template beat;
 * rr: (STAGE_N_LAGS,) normalised autocorrelation of |p| - mean. Returns the number of beats
 * that built the template. */
int ecg_stage_run(const float *x, const float *p, float *residual, float *qrs, float *rr);

#endif
