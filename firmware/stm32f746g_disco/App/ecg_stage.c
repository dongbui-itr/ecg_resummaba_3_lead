#include "ecg_stage.h"

#include <math.h>
#include <string.h>

#define T STAGE_STEPS
#define C STAGE_LEADS
#define L (STAGE_PRE + STAGE_POST + 1)

static float xd[T * C];
static float w[T];
static float tpl[L * C];
static int beat_at[T];

int ecg_stage_run(const float *x, const float *p, float *residual, float *qrs, float *rr)
{
    /* 125 Hz copy of the input (avg_pool 2) */
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            xd[t * C + c] = 0.5f * (x[(2 * t) * C + c] + x[(2 * t + 1) * C + c]);

    /* beats = local maxima of p over +-STAGE_NMS/2 steps that reach STAGE_MIN_PROB */
    const int half = STAGE_NMS / 2;
    int n_beats = 0;
    float wsum = 0.0f;
    for (int t = 0; t < T; t++) {
        float peak = p[t];
        int a = t - half < 0 ? 0 : t - half, b = t + half >= T ? T - 1 : t + half;
        for (int i = a; i <= b; i++)
            if (p[i] > peak)
                peak = p[i];
        w[t] = (p[t] >= peak && p[t] >= STAGE_MIN_PROB) ? p[t] : 0.0f;
        if (w[t] > 0.0f) {
            beat_at[n_beats++] = t;
            wsum += w[t];
        }
    }

    /* template[l] = sum_t w[t] xd[t + l - PRE] / sum w  (zero outside the window) */
    memset(tpl, 0, sizeof(tpl));
    for (int k = 0; k < n_beats; k++) {
        int s = beat_at[k];
        for (int l = 0; l < L; l++) {
            int i = s + l - STAGE_PRE;
            if (i < 0 || i >= T)
                continue;
            for (int c = 0; c < C; c++)
                tpl[l * C + c] += w[s] * xd[i * C + c];
        }
    }
    const float norm = 1.0f / (wsum + 1e-6f);
    for (int i = 0; i < L * C; i++)
        tpl[i] *= norm;

    /* residual = xd - sum_s w[s] template[t - s + PRE] */
    memcpy(residual, xd, sizeof(xd));
    for (int k = 0; k < n_beats; k++) {
        int s = beat_at[k];
        for (int l = 0; l < L; l++) {
            int t = s + l - STAGE_PRE;
            if (t < 0 || t >= T)
                continue;
            for (int c = 0; c < C; c++)
                residual[t * C + c] -= w[s] * tpl[l * C + c];
        }
    }

    /* qrs = 1 within +-STAGE_MASK/2 steps of a template beat */
    const int mh = STAGE_MASK / 2;
    memset(qrs, 0, T * sizeof(float));
    for (int k = 0; k < n_beats; k++)
        for (int t = beat_at[k] - mh; t <= beat_at[k] + mh; t++)
            if (t >= 0 && t < T)
                qrs[t] = 1.0f;

    /* RR descriptor: e = |p| - mean, rr[j] = mean(base * e[lag:]) / (mean(base^2) + 1e-6),
     * base = e[0:span], lag = RR_LO + j */
    static float e[T];
    float mean = 0.0f;
    for (int t = 0; t < T; t++) {
        e[t] = fabsf(p[t]);
        mean += e[t];
    }
    mean /= (float)T;
    for (int t = 0; t < T; t++)
        e[t] -= mean;
    const int span = T - STAGE_RR_HI;
    float den = 0.0f;
    for (int t = 0; t < span; t++)
        den += e[t] * e[t];
    den = den / (float)span + 1e-6f;
    for (int j = 0; j < STAGE_N_LAGS; j++) {
        const float *sh = e + STAGE_RR_LO + j;
        float num = 0.0f;
        for (int t = 0; t < span; t++)
            num += e[t] * sh[t];
        rr[j] = num / (float)span / den;
    }
    return n_beats;
}
