/*
 * Model output -> what the kit reports, mirroring the Python pipeline:
 *   per-second rhythm probabilities  = mean of the 125 rows of each second (predict.py at 1 Hz)
 *   beats                            = ecgr/rhythm/beats.py pick_beats (same threshold and
 *                                      refractory as rc.BEAT_PICK_THRESHOLD / _REFRACTORY_SECONDS)
 *   heart rate                       = 60 / median R-R of the picked beats
 * The overlap between consecutive windows is merged in ecg_app.c (mean of both windows, as
 * predict.py does with the taper off).
 */
#ifndef ECG_POST_H
#define ECG_POST_H

#include <stdint.h>

#include "ecg_model.h"

#define POST_STEPS_PER_SEC   125
#define POST_SECONDS         10
#define POST_BEAT_THRESHOLD  0.5f   /* rc.BEAT_PICK_THRESHOLD */
#define POST_REFRACTORY      25     /* round(rc.BEAT_REFRACTORY_SECONDS 0.2 s * 125 Hz) */
#define POST_MAX_BEATS       64

typedef struct {
    uint16_t pos;     /* sample at 250 Hz inside the window (step * 2) */
    uint8_t cls;      /* 1 N, 2 S, 3 V (reference beats may also be 4 = other) */
    uint8_t conf;     /* round(255 * smoothed p(beat)); 0 for reference beats */
} ecg_beat_t;

/* rhythm (1250, 5) -> out[10][5] */
void post_seconds(const float *rhythm, float out[POST_SECONDS][MODEL_CLASSES]);
/* beat (1250, 4) -> beats sorted by position; returns the count (<= max) */
int post_pick_beats(const float *beat, ecg_beat_t *out, int max);
/* beats -> bpm * 10 (0 with fewer than 2 beats) */
uint16_t post_heart_rate_x10(const ecg_beat_t *beats, int n);

#endif
