#include "ecg_post.h"

#include <string.h>

#define T MODEL_STEPS

void post_seconds(const float *rhythm, float out[POST_SECONDS][MODEL_CLASSES])
{
    for (int s = 0; s < POST_SECONDS; s++) {
        float acc[MODEL_CLASSES] = {0};
        for (int i = s * POST_STEPS_PER_SEC; i < (s + 1) * POST_STEPS_PER_SEC; i++)
            for (int c = 0; c < MODEL_CLASSES; c++)
                acc[c] += rhythm[i * MODEL_CLASSES + c];
        for (int c = 0; c < MODEL_CLASSES; c++)
            out[s][c] = acc[c] / (float)POST_STEPS_PER_SEC;
    }
}

static float ps[T];
static int cand[T];
static uint8_t taken[T];

int post_pick_beats(const float *beat, ecg_beat_t *out, int max)
{
    /* p(beat) = 1 - p(none), smoothed over 3 steps (np.convolve 'same', zero ends) */
    for (int i = 0; i < T; i++) {
        float prev = i > 0 ? 1.0f - beat[(i - 1) * 4] : 0.0f;
        float next = i < T - 1 ? 1.0f - beat[(i + 1) * 4] : 0.0f;
        ps[i] = (prev + (1.0f - beat[i * 4]) + next) / 3.0f;
    }
    int n = 0;
    for (int i = 1; i < T - 1; i++)
        if (ps[i] >= ps[i - 1] && ps[i] > ps[i + 1] && ps[i] > POST_BEAT_THRESHOLD)
            cand[n++] = i;
    /* strongest first, ties by position (np.argsort(-ps, kind='stable')) */
    for (int a = 1; a < n; a++) {
        int v = cand[a], b = a - 1;
        while (b >= 0 && ps[cand[b]] < ps[v]) {
            cand[b + 1] = cand[b];
            b--;
        }
        cand[b + 1] = v;
    }
    memset(taken, 0, sizeof(taken));
    int kept = 0;
    for (int k = 0; k < n && kept < max; k++) {
        int i = cand[k];
        if (taken[i])
            continue;
        int a = i - POST_REFRACTORY < 0 ? 0 : i - POST_REFRACTORY;
        int b = i + POST_REFRACTORY >= T ? T - 1 : i + POST_REFRACTORY;
        memset(taken + a, 1, (size_t)(b - a + 1));
        /* class = argmax of N / S / V averaged over +-2 steps */
        float m[3] = {0};
        int lo = i - 2 < 0 ? 0 : i - 2, hi = i + 3 > T ? T : i + 3;
        for (int j = lo; j < hi; j++)
            for (int c = 0; c < 3; c++)
                m[c] += beat[j * 4 + 1 + c];
        int cls = 0;
        for (int c = 1; c < 3; c++)
            if (m[c] > m[cls])
                cls = c;
        float conf = ps[i] * 255.0f + 0.5f;
        out[kept].pos = (uint16_t)(i * 2);
        out[kept].cls = (uint8_t)(1 + cls);
        out[kept].conf = (uint8_t)(conf > 255.0f ? 255.0f : conf);
        kept++;
    }
    /* by position */
    for (int a = 1; a < kept; a++) {
        ecg_beat_t v = out[a];
        int b = a - 1;
        while (b >= 0 && out[b].pos > v.pos) {
            out[b + 1] = out[b];
            b--;
        }
        out[b + 1] = v;
    }
    return kept;
}

uint16_t post_heart_rate_x10(const ecg_beat_t *beats, int n)
{
    if (n < 2)
        return 0;
    uint16_t rr[POST_MAX_BEATS];
    int m = 0;
    for (int k = 1; k < n && m < POST_MAX_BEATS; k++)
        rr[m++] = (uint16_t)(beats[k].pos - beats[k - 1].pos);
    for (int a = 1; a < m; a++) {
        uint16_t v = rr[a];
        int b = a - 1;
        while (b >= 0 && rr[b] > v) {
            rr[b + 1] = rr[b];
            b--;
        }
        rr[b + 1] = v;
    }
    float med = (m % 2) ? rr[m / 2] : 0.5f * (rr[m / 2 - 1] + rr[m / 2]);
    if (med <= 0.0f)
        return 0;
    return (uint16_t)(600.0f * 250.0f / med + 0.5f);      /* 60 s * 10 / (med / 250 Hz) */
}
