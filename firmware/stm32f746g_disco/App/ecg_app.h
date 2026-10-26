/*
 * The kit's application: receive ECG chunks from the PC, keep a 10 s window that advances
 * 9 s per chunk (1 s overlap), run the model, send the result back and draw it on the LCD.
 *
 *   chunk 0     samples [0, 2500)                         -> window 0 = [0 s, 10 s)
 *   chunk k>=1  the next 2250 samples (9 s)               -> window k = [9k s, 9k + 10 s)
 *
 * Second 0 of window k (k >= 1) is the last second of window k-1: its rhythm probabilities
 * and beat rows are the mean of both windows (ecgr/rhythm/predict.py, taper off), so the
 * PC can stitch windows into one record: seconds [9k, 9k+9) of window k are final, its last
 * second is replaced by the next window's (merged) first second.
 */
#ifndef ECG_APP_H
#define ECG_APP_H

#include <stdint.h>

#include "ecg_model.h"
#include "ecg_post.h"

#define APP_FS            250
#define APP_WIN_SAMPLES   2500
#define APP_HOP_SAMPLES   2250
#define APP_WIN_SECONDS   10
#define APP_OVERLAP_SEC   1
#define LABEL_NONE        0xFF          /* no reference rhythm for that second */

#ifndef APP_BAUD
#define APP_BAUD          921600        /* USART1 = ST-LINK virtual COM port; make APP_BAUD=... */
#endif
#define APP_STR_(v)       #v
#define APP_STR(v)        APP_STR_(v)
#define APP_BAUD_TEXT     APP_STR(APP_BAUD)

typedef struct {
    uint32_t index;                     /* chunk / window number */
    uint32_t win_start;                 /* first sample of the window in the stream (250 Hz) */
    uint32_t us_infer, us_total;        /* net A + stage + net B; everything incl. drawing */
    uint32_t us_a, us_stage, us_b;
    float sec_prob[POST_SECONDS][MODEL_CLASSES];
    uint8_t pred[POST_SECONDS];         /* argmax of sec_prob */
    uint8_t ref[POST_SECONDS];          /* label from the PC, LABEL_NONE if unknown */
    float p_noise[MODEL_SEGMENTS];
    uint8_t lead[MODEL_SEGMENTS];       /* argmax of the 'channel' row: 0 NOISE, 1..3 CH */
    uint16_t hr_x10;
    int n_beats, n_ref_beats;
    ecg_beat_t beats[POST_MAX_BEATS];
    ecg_beat_t ref_beats[POST_MAX_BEATS];
} ecg_result_t;

extern const char *const APP_CLASS_NAMES[MODEL_CLASSES];

/* act_a / act_b: activation buffers for the two nets. Returns 0 when the model is ready. */
int app_init(void *act_a, void *act_b);
/* Bytes from the PC link; complete frames are handled (inference included) before returning. */
void app_rx(const uint8_t *data, uint32_t n);

#endif
