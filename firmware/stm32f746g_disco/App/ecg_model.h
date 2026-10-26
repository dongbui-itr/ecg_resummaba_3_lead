/*
 * The rhythm model for one 10 s window: net A -> ecg_stage -> net B (X-CUBE-AI legacy API).
 * Same file on the board and in the PC's virtual kit (host/): both link the generated
 * Model/ecga*.c, Model/ecgb*.c against the X-CUBE-AI runtime of their CPU.
 */
#ifndef ECG_MODEL_H
#define ECG_MODEL_H

#include <stdint.h>

#define MODEL_SAMPLES   2500        /* 10 s at 250 Hz */
#define MODEL_LEADS     3
#define MODEL_STEPS     1250        /* output rows: 8 ms */
#define MODEL_CLASSES   5           /* SINUS AFIB SVT VT AVB */
#define MODEL_BEAT_COLS 4           /* none N S V */
#define MODEL_SEGMENTS  5           /* 'channel' rows: 2 s each */
#define MODEL_CHANNELS  4           /* NOISE CH1 CH2 CH3 */

typedef struct {
    const float *rhythm;    /* (1250, 5) softmax */
    const float *beat;      /* (1250, 4) none / N / S / V */
    const float *channel;   /* (5, 4) NOISE / CH1..3 */
    uint32_t us_a, us_stage, us_b;
} ecg_model_out_t;

/* act_a / act_b: activation buffers of ECG_MODEL_ACT_A / ECG_MODEL_ACT_B bytes (SDRAM on
 * the board). Returns 0 on success. */
int ecg_model_init(void *act_a, void *act_b);
/* x: (2500, 3) z-scored window. The outputs point into net B's activations and stay valid
 * until the next call. Returns 0 on success. */
int ecg_model_run(const float *x, ecg_model_out_t *out);

#endif
