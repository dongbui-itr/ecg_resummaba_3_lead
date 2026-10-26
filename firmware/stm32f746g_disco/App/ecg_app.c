#include "ecg_app.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ecg_proto.h"
#include "ecg_view.h"
#include "model_meta.h"
#include "platform.h"

const char *const APP_CLASS_NAMES[MODEL_CLASSES] = {"SINUS", "AFIB", "SVT", "VT", "AVB"};

#define BEAT_ROWS_PER_SEC POST_STEPS_PER_SEC

static proto_parser_t parser;
static uint8_t tx[1024 + PROTO_OVERHEAD];
static uint8_t tx_seq;

static struct {
    int ready;                          /* the model initialised */
    int have_window;
    uint32_t next_index;
    int16_t win[APP_WIN_SAMPLES * MODEL_LEADS];        /* microvolts */
    uint8_t ref[POST_SECONDS];
    int n_ref_beats;
    ecg_beat_t ref_beats[POST_MAX_BEATS];
    /* the previous window's last second, for the overlap */
    int have_prev;
    float prev_sec[MODEL_CLASSES];
    float prev_beat[BEAT_ROWS_PER_SEC * MODEL_BEAT_COLS];
} st;

static float x[APP_WIN_SAMPLES * MODEL_LEADS];
static float beat_merged[MODEL_STEPS * MODEL_BEAT_COLS];
static ecg_result_t res;

static void send(uint8_t type, const uint8_t *payload, uint16_t len)
{
    size_t n = proto_encode(tx, type, tx_seq++, payload, len);
    plat_uart_write(tx, n);
}

static void send_error(uint8_t code, uint32_t index, const char *text)
{
    uint8_t buf[128], *p = buf;
    p = wr_u8(p, code);
    p = wr_u32(p, index);
    size_t n = strlen(text);
    if (n > sizeof(buf) - 5)
        n = sizeof(buf) - 5;
    memcpy(p, text, n);
    send(MSG_ERROR, buf, (uint16_t)(5 + n));
}

static void send_info(void)
{
    uint8_t buf[80], *p = buf;
    memset(buf, 0, sizeof(buf));
    p = wr_u8(p, PROTO_VERSION);
    p = wr_u8(p, MODEL_CLASSES);
    p = wr_u16(p, APP_FS);
    p = wr_u16(p, APP_WIN_SAMPLES);
    p = wr_u16(p, APP_HOP_SAMPLES);
    p = wr_u32(p, ECG_MODEL_MACC);
    p = wr_u32(p, ECG_MODEL_ACT_A + ECG_MODEL_ACT_B);
    p = wr_u32(p, ECG_MODEL_WEIGHTS);
    p = wr_u32(p, plat_cpu_hz());
    strncpy((char *)p, ECG_MODEL_TAG, 23);
    p += 24;
    strncpy((char *)p, __DATE__ " " __TIME__, 23);
    p += 24;
    send(MSG_INFO, buf, (uint16_t)(p - buf));
}

static uint8_t u8p(float v)
{
    float q = v * 255.0f + 0.5f;
    return (uint8_t)(q < 0.0f ? 0.0f : (q > 255.0f ? 255.0f : q));
}

static void send_result(const ecg_result_t *r)
{
    uint8_t buf[1024], *p = buf;
    p = wr_u32(p, r->index);
    p = wr_u32(p, r->win_start);
    p = wr_u32(p, r->us_infer);
    p = wr_u32(p, r->us_total);
    p = wr_u8(p, POST_SECONDS);
    for (int s = 0; s < POST_SECONDS; s++) {
        p = wr_u8(p, r->pred[s]);
        p = wr_u8(p, r->ref[s]);
        for (int c = 0; c < MODEL_CLASSES; c++)
            p = wr_u8(p, u8p(r->sec_prob[s][c]));
    }
    for (int g = 0; g < MODEL_SEGMENTS; g++)
        p = wr_u8(p, u8p(r->p_noise[g]));
    for (int g = 0; g < MODEL_SEGMENTS; g++)
        p = wr_u8(p, r->lead[g]);
    p = wr_u16(p, r->hr_x10);
    p = wr_u16(p, (uint16_t)r->n_beats);
    for (int k = 0; k < r->n_beats; k++) {
        p = wr_u16(p, r->beats[k].pos);
        p = wr_u8(p, r->beats[k].cls);
        p = wr_u8(p, r->beats[k].conf);
    }
    p = wr_u32(p, r->us_a);
    p = wr_u32(p, r->us_stage);
    p = wr_u32(p, r->us_b);
    send(MSG_RESULT, buf, (uint16_t)(p - buf));
}

/* Per-lead z-score of the window (signal_ops.normalize_window), microvolts -> float. */
static void normalise(void)
{
    for (int c = 0; c < MODEL_LEADS; c++) {
        double sum = 0.0, sq = 0.0;
        for (int i = 0; i < APP_WIN_SAMPLES; i++) {
            double v = st.win[i * MODEL_LEADS + c] * 1e-3;
            sum += v;
            sq += v * v;
        }
        double mean = sum / APP_WIN_SAMPLES;
        double var = sq / APP_WIN_SAMPLES - mean * mean;
        double inv = var > 1e-18 ? 1.0 / sqrt(var) : 1.0;
        for (int i = 0; i < APP_WIN_SAMPLES; i++)
            x[i * MODEL_LEADS + c] = (float)((st.win[i * MODEL_LEADS + c] * 1e-3 - mean) * inv);
    }
}

static void process(uint32_t index, uint32_t t_start)
{
    ecg_model_out_t out;
    normalise();
    if (ecg_model_run(x, &out) != 0) {
        send_error(ERR_MODEL, index, "model run failed");
        return;
    }
    memset(&res, 0, sizeof(res));
    res.index = index;
    res.win_start = index * APP_HOP_SAMPLES;
    res.us_a = out.us_a;
    res.us_stage = out.us_stage;
    res.us_b = out.us_b;
    res.us_infer = out.us_a + out.us_stage + out.us_b;

    post_seconds(out.rhythm, res.sec_prob);
    memcpy(beat_merged, out.beat, sizeof(beat_merged));
    float last_sec[MODEL_CLASSES];
    memcpy(last_sec, res.sec_prob[POST_SECONDS - 1], sizeof(last_sec));
    if (index > 0 && st.have_prev) {
        for (int c = 0; c < MODEL_CLASSES; c++)
            res.sec_prob[0][c] = 0.5f * (res.sec_prob[0][c] + st.prev_sec[c]);
        for (int i = 0; i < BEAT_ROWS_PER_SEC * MODEL_BEAT_COLS; i++)
            beat_merged[i] = 0.5f * (beat_merged[i] + st.prev_beat[i]);
    }
    /* this window's own last second, for the next window */
    memcpy(st.prev_sec, last_sec, sizeof(last_sec));
    memcpy(st.prev_beat, out.beat + (MODEL_STEPS - BEAT_ROWS_PER_SEC) * MODEL_BEAT_COLS,
           sizeof(st.prev_beat));
    st.have_prev = 1;

    for (int s = 0; s < POST_SECONDS; s++) {
        int best = 0;
        for (int c = 1; c < MODEL_CLASSES; c++)
            if (res.sec_prob[s][c] > res.sec_prob[s][best])
                best = c;
        res.pred[s] = (uint8_t)best;
        res.ref[s] = st.ref[s];
    }
    for (int g = 0; g < MODEL_SEGMENTS; g++) {
        const float *row = out.channel + g * MODEL_CHANNELS;
        int best = 0;
        for (int c = 1; c < MODEL_CHANNELS; c++)
            if (row[c] > row[best])
                best = c;
        res.p_noise[g] = row[0];
        res.lead[g] = (uint8_t)best;
    }
    res.n_beats = post_pick_beats(beat_merged, res.beats, POST_MAX_BEATS);
    res.hr_x10 = post_heart_rate_x10(res.beats, res.n_beats);
    res.n_ref_beats = st.n_ref_beats;
    memcpy(res.ref_beats, st.ref_beats, sizeof(res.ref_beats));

    ecg_view_render(st.win, &res);
    res.us_total = plat_micros() - t_start;
    send_result(&res);
}

static void handle_chunk(const proto_frame_t *f)
{
    uint32_t t_start = plat_micros();
    const uint8_t *p = f->payload;
    if (f->len < 10) {
        send_error(ERR_CHUNK_SIZE, 0, "short chunk");
        return;
    }
    uint32_t index = rd_u32(p);
    uint16_t n = rd_u16(p + 4);
    uint8_t leads = p[6], nsec = p[7];
    uint16_t nref = rd_u16(p + 8);
    uint32_t need = 10u + (uint32_t)n * leads * 2u + nsec + nref * 3u;
    uint16_t want = index == 0 ? APP_WIN_SAMPLES : APP_HOP_SAMPLES;
    if (leads != MODEL_LEADS || n != want || nsec != n / APP_FS || f->len != need) {
        char msg[96];
        snprintf(msg, sizeof(msg), "chunk %lu: %u samples x %u leads, %u s, len %u (need %u x 3)",
                 (unsigned long)index, n, leads, nsec, f->len, want);
        send_error(ERR_CHUNK_SIZE, index, msg);
        return;
    }
    if (index != 0 && (!st.have_window || index != st.next_index)) {
        char msg[64];
        snprintf(msg, sizeof(msg), "expected chunk %lu, got %lu", (unsigned long)st.next_index,
                 (unsigned long)index);
        send_error(ERR_SEQUENCE, index, msg);
        return;
    }
    const uint8_t *samples = p + 10;
    const uint8_t *labels = samples + (uint32_t)n * leads * 2u;
    const uint8_t *beats = labels + nsec;

    int keep = 0;                                  /* samples kept from the previous window */
    if (index == 0) {
        st.have_prev = 0;
        st.n_ref_beats = 0;
    } else {
        keep = APP_WIN_SAMPLES - APP_HOP_SAMPLES;
        memmove(st.win, st.win + APP_HOP_SAMPLES * MODEL_LEADS, (size_t)keep * MODEL_LEADS * 2);
        memmove(st.ref, st.ref + (POST_SECONDS - APP_OVERLAP_SEC), APP_OVERLAP_SEC);
        int m = 0;
        for (int k = 0; k < st.n_ref_beats; k++)
            if (st.ref_beats[k].pos >= APP_HOP_SAMPLES) {
                st.ref_beats[m] = st.ref_beats[k];
                st.ref_beats[m++].pos -= APP_HOP_SAMPLES;
            }
        st.n_ref_beats = m;
    }
    for (uint32_t i = 0; i < (uint32_t)n * leads; i++)
        st.win[keep * MODEL_LEADS + i] = (int16_t)rd_u16(samples + 2 * i);
    memcpy(st.ref + keep / APP_FS, labels, nsec);
    for (int k = 0; k < nref && st.n_ref_beats < POST_MAX_BEATS; k++) {
        ecg_beat_t *b = &st.ref_beats[st.n_ref_beats++];
        b->pos = (uint16_t)(rd_u16(beats + 3 * k) + keep);
        b->cls = beats[3 * k + 2];
        b->conf = 0;
    }
    st.have_window = 1;
    st.next_index = index + 1;
    process(index, t_start);
}

static void handle(const proto_frame_t *f)
{
    switch (f->type) {
    case MSG_HELLO:
        st.have_window = 0;
        st.have_prev = 0;
        st.next_index = 0;
        send_info();
        ecg_view_status("PC connected", "waiting for the first 10 s chunk");
        break;
    case MSG_CHUNK:
        if (!st.ready)
            send_error(ERR_MODEL, 0, "model not initialised");
        else
            handle_chunk(f);
        break;
    default:
        send_error(ERR_UNKNOWN_TYPE, 0, "unknown message type");
    }
}

int app_init(void *act_a, void *act_b)
{
    proto_parser_init(&parser);
    memset(&st, 0, sizeof(st));
    int rc = ecg_model_init(act_a, act_b);
    st.ready = rc == 0;
    ecg_view_init();
    if (st.ready)
        ecg_view_status("Waiting for the PC", "USART1 (ST-LINK VCP) " APP_BAUD_TEXT " 8N1");
    else
        ecg_view_status("Model init FAILED", "check the activation buffers");
    return rc;
}

void app_rx(const uint8_t *data, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        if (proto_feed(&parser, data[i]))
            handle(&parser.frame);
}
