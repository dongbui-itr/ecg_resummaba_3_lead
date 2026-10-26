#include "ecg_view.h"

#include <stdio.h>
#include <string.h>

#include "model_meta.h"
#include "platform.h"

extern sFONT Font8, Font12;

#define C_HDR_BG     RGB(0x1F2430)
#define C_HDR_FG     RGB(0xFFFFFF)
#define C_HDR_DIM    RGB(0xAAB2C0)
#define C_PAPER      RGB(0xFFF6F4)
#define C_MINOR      RGB(0xF9DEDA)
#define C_MAJOR      RGB(0xE8A39B)
#define C_PANEL      RGB(0xF3F4F6)
#define C_PANEL_LINE RGB(0xD1D5DB)
#define C_TRACE      RGB(0x111111)
#define C_FLAT       RGB(0x9CA3AF)
#define C_TEXT       RGB(0x1F2937)
#define C_MUTED      RGB(0x6B7280)
#define C_UNKNOWN    RGB(0xC9CDD3)
#define C_MISMATCH   RGB(0xDC2626)
#define C_WARN       RGB(0xFACC15)

/* SINUS AFIB SVT VT AVB - same colours as the web viewer */
static const uint32_t CLASS_RGB[MODEL_CLASSES] = {0x4D9A6A, 0xE08214, 0x2B83BA, 0xD7191C, 0x7B3294};
static const char *const CLASS_SHORT[MODEL_CLASSES] = {"SR", "AF", "SVT", "VT", "AVB"};
/* beat classes 1 N, 2 S, 3 V, 4 other */
static const uint32_t BEAT_RGB[5] = {0x6B7280, 0x1E7B3A, 0x1F5FAF, 0xC81E1E, 0x6B7280};
static const char BEAT_CHAR[5] = {'?', 'N', 'S', 'V', 'Q'};

static void text_center(gfx_canvas_t *c, int cx, int y, const char *s, const sFONT *f, uint32_t argb)
{
    gfx_text(c, cx - gfx_text_width(s, f) / 2, y, s, f, argb);
}

/* --- layer 1: ECG paper ------------------------------------------------------------------ */

static void draw_grid(void)
{
    gfx_canvas_t *g = plat_layer(LAYER_GRID);
    gfx_clear(g, C_PANEL);
    gfx_fill_rect(g, 0, 0, LCD_W, VIEW_HDR_H, C_HDR_BG);
    gfx_fill_rect(g, 0, VIEW_ECG_Y, LCD_W, VIEW_ECG_H, C_PAPER);
    /* 1 mm = 0.04 s = 480 / 250 px; 250 mm across, ~100 mm down */
    for (int pass = 0; pass < 2; pass++) {
        uint32_t col = pass ? C_MAJOR : C_MINOR;
        for (int i = 0; i <= 250; i++) {
            if (pass != (i % 5 == 0))
                continue;
            int x = (i * LCD_W + 125) / 250;
            gfx_vline(g, x >= LCD_W ? LCD_W - 1 : x, VIEW_ECG_Y, VIEW_ECG_H, col);
        }
        for (int j = 0;; j++) {
            int y = VIEW_ECG_Y + (j * LCD_W * 4 + 500) / 1000;      /* j * 1.92 px */
            if (y >= VIEW_ECG_Y + VIEW_ECG_H)
                break;
            if (pass == (j % 5 == 0))
                gfx_hline(g, 0, y, LCD_W, col);
        }
    }
    gfx_hline(g, 0, VIEW_RES_Y, LCD_W, C_PANEL_LINE);
}

/* --- layer 2: signal --------------------------------------------------------------------- */

static void draw_signal(const int16_t *win)
{
    gfx_canvas_t *c = plat_layer(LAYER_SIGNAL);
    gfx_clear(c, TRANSPARENT);
    static const char *const names[MODEL_LEADS] = {"CH1", "CH2", "CH3"};
    for (int l = 0; l < MODEL_LEADS; l++) {
        long sum = 0;
        int flat = 1;
        for (int i = 0; i < APP_WIN_SAMPLES; i++) {
            int16_t v = win[i * MODEL_LEADS + l];
            sum += v;
            if (v != win[l])
                flat = 0;
        }
        float mean = (float)sum / APP_WIN_SAMPLES;
        int yc = VIEW_LANE_Y + l * VIEW_LANE_H + VIEW_LANE_H / 2;
        int top = VIEW_ECG_Y, bot = VIEW_ECG_Y + VIEW_ECG_H - 1;
        if (flat) {
            gfx_hline(c, 0, yc, LCD_W, C_FLAT);
            gfx_text(c, 40, yc - 12, "no signal", &Font8, C_MUTED);
        } else {
            int prev_y = -1;
            for (int x = 0; x < LCD_W; x++) {
                int a = x * APP_WIN_SAMPLES / LCD_W, b = (x + 1) * APP_WIN_SAMPLES / LCD_W;
                int lo = 1 << 30, hi = -(1 << 30);
                for (int i = a; i < b; i++) {
                    float mv = (win[i * MODEL_LEADS + l] - mean) * 1e-3f;
                    int y = yc - (int)(mv * VIEW_PX_PER_MV + (mv >= 0 ? 0.5f : -0.5f));
                    if (y < lo) lo = y;
                    if (y > hi) hi = y;
                }
                if (prev_y >= 0) {            /* join the previous column */
                    if (prev_y < lo) lo = prev_y;
                    if (prev_y > hi) hi = prev_y;
                }
                int y_last;
                {
                    float mv = (win[(b - 1) * MODEL_LEADS + l] - mean) * 1e-3f;
                    y_last = yc - (int)(mv * VIEW_PX_PER_MV + (mv >= 0 ? 0.5f : -0.5f));
                }
                prev_y = y_last;
                if (lo < top) lo = top;
                if (hi > bot) hi = bot;
                if (hi >= lo)
                    gfx_vline(c, x, lo, hi - lo + 1, C_TRACE);
            }
        }
        int ly = VIEW_LANE_Y + l * VIEW_LANE_H + 2;
        gfx_fill_rect(c, 1, ly - 1, 24, 14, ARGB(0xC0, 0xFF, 0xFF, 0xFF));
        gfx_text(c, 3, ly, names[l], &Font12, C_TEXT);
    }
}

/* --- layer 3: result --------------------------------------------------------------------- */

static void fmt_time(char *out, size_t n, uint32_t seconds)
{
    snprintf(out, n, "%lu:%02lu", (unsigned long)(seconds / 60), (unsigned long)(seconds % 60));
}

static int ref_has(const ecg_result_t *r, const ecg_beat_t *b)
{
    for (int k = 0; k < r->n_ref_beats; k++) {
        int d = (int)r->ref_beats[k].pos - (int)b->pos;
        if (d >= -37 && d <= 37 && r->ref_beats[k].cls == b->cls)    /* +-150 ms, same class */
            return 1;
    }
    return 0;
}

static void draw_header(gfx_canvas_t *c, const ecg_result_t *r)
{
    char buf[64], t0[12], t1[12];
    gfx_text(c, 4, 4, "ECG AI " ECG_MODEL_TAG, &Font12, C_HDR_FG);
    if (!r)
        return;
    fmt_time(t0, sizeof(t0), r->win_start / APP_FS);
    fmt_time(t1, sizeof(t1), r->win_start / APP_FS + APP_WIN_SECONDS);
    snprintf(buf, sizeof(buf), "#%lu %s-%s", (unsigned long)r->index, t0, t1);
    text_center(c, LCD_W / 2, 4, buf, &Font12, C_HDR_FG);
    snprintf(buf, sizeof(buf), "AI %lu ms", (unsigned long)((r->us_infer + 500) / 1000));
    gfx_text(c, LCD_W - 4 - gfx_text_width(buf, &Font12), 4, buf, &Font12, C_HDR_DIM);
}

static void draw_result(const ecg_result_t *r)
{
    gfx_canvas_t *c = plat_layer(LAYER_RESULT);
    gfx_clear(c, TRANSPARENT);
    draw_header(c, r);
    const int ecg_bot = VIEW_ECG_Y + VIEW_ECG_H;

    /* non-sinus seconds: a faint band of the class colour over the leads */
    for (int s = 0; s < POST_SECONDS; s++)
        if (r->pred[s] != 0)
            gfx_fill_rect(c, s * VIEW_PX_PER_S, VIEW_ECG_Y, VIEW_PX_PER_S, VIEW_ECG_H,
                          ARGB(0x38, 0, 0, 0) | CLASS_RGB[r->pred[s]]);
    /* unreadable 2 s segments */
    for (int g = 0; g < MODEL_SEGMENTS; g++)
        if (r->p_noise[g] >= 0.5f) {
            int x = g * 2 * VIEW_PX_PER_S;
            gfx_fill_rect(c, x, VIEW_ECG_Y, 2 * VIEW_PX_PER_S, VIEW_ECG_H, ARGB(0x50, 0x80, 0x80, 0x80));
            gfx_text(c, x + 3, ecg_bot - 22, "NOISE", &Font8, C_TEXT);
        }
    /* the second shared with the previous window */
    if (r->index > 0) {
        gfx_dashed_vline(c, APP_OVERLAP_SEC * VIEW_PX_PER_S, VIEW_ECG_Y, VIEW_ECG_H, 4, 3, C_MUTED);
        gfx_text(c, 3, ecg_bot - 10, "overlap", &Font8, C_MUTED);
    }
    /* beats: AI letter boxes on top (yellow = no reference beat of that class within
     * 150 ms), reference letters under them */
    for (int k = 0; k < r->n_beats; k++) {
        const ecg_beat_t *b = &r->beats[k];
        int x = b->pos * LCD_W / APP_WIN_SAMPLES;
        int miss = r->n_ref_beats > 0 && !ref_has(r, b);
        uint32_t box = miss ? C_WARN : (0xFF000000u | BEAT_RGB[b->cls]);
        gfx_fill_rect(c, x - 3, VIEW_ECG_Y + 1, 7, 10, box);
        char ch[2] = {BEAT_CHAR[b->cls], 0};
        gfx_text(c, x - 2, VIEW_ECG_Y + 2, ch, &Font8, miss ? C_TEXT : C_HDR_FG);
    }
    for (int k = 0; k < r->n_ref_beats; k++) {
        const ecg_beat_t *b = &r->ref_beats[k];
        int x = b->pos * LCD_W / APP_WIN_SAMPLES;
        char ch[2] = {BEAT_CHAR[b->cls < 5 ? b->cls : 0], 0};
        gfx_text(c, x - 2, VIEW_ECG_Y + 13, ch, &Font8, 0xFF000000u | BEAT_RGB[b->cls < 5 ? b->cls : 0]);
    }

    /* per-second rows: reference (top) and AI (bottom) */
    char buf[96];
    int match = 0, known = 0;
    for (int s = 0; s < POST_SECONDS; s++) {
        int x = s * VIEW_PX_PER_S + 1, w = VIEW_PX_PER_S - 2;
        uint8_t ref = r->ref[s];
        if (ref < MODEL_CLASSES) {
            gfx_fill_rect(c, x, VIEW_RES_Y + 2, w, 14, 0xFF000000u | CLASS_RGB[ref]);
            text_center(c, x + w / 2, VIEW_RES_Y + 3, CLASS_SHORT[ref], &Font12, C_HDR_FG);
            known++;
            match += ref == r->pred[s];
        } else {
            gfx_fill_rect(c, x, VIEW_RES_Y + 2, w, 14, C_UNKNOWN);
            text_center(c, x + w / 2, VIEW_RES_Y + 3, "--", &Font12, C_TEXT);
        }
        uint8_t p = r->pred[s];
        int pct = (int)(r->sec_prob[s][p] * 100.0f + 0.5f);
        snprintf(buf, sizeof(buf), "%s %d", CLASS_SHORT[p], pct > 99 ? 99 : pct);
        gfx_fill_rect(c, x, VIEW_RES_Y + 18, w, 14, 0xFF000000u | CLASS_RGB[p]);
        text_center(c, x + w / 2, VIEW_RES_Y + 19, buf, &Font12, C_HDR_FG);
        if (ref < MODEL_CLASSES && ref != p)
            gfx_fill_rect(c, x, VIEW_RES_Y + 33, w, 2, C_MISMATCH);
    }

    /* summary */
    int cnt[5] = {0}, rcnt[5] = {0};
    for (int k = 0; k < r->n_beats; k++)
        cnt[r->beats[k].cls < 5 ? r->beats[k].cls : 0]++;
    for (int k = 0; k < r->n_ref_beats; k++)
        rcnt[r->ref_beats[k].cls < 5 ? r->ref_beats[k].cls : 0]++;
    snprintf(buf, sizeof(buf), "HR %u bpm  AI N%d S%d V%d  REF N%d S%d V%d",
             (unsigned)((r->hr_x10 + 5) / 10), cnt[1], cnt[2], cnt[3], rcnt[1], rcnt[2], rcnt[3]);
    gfx_text(c, 4, VIEW_RES_Y + 37, buf, &Font12, C_TEXT);
    int noisy = 0;
    for (int g = 0; g < MODEL_SEGMENTS; g++)
        noisy += r->p_noise[g] >= 0.5f;
    if (known)
        snprintf(buf, sizeof(buf), "rhythm AI=REF %d/%d s  noise %d/5  top REF / bottom AI", match, known, noisy);
    else
        snprintf(buf, sizeof(buf), "no reference labels  noise %d/5  25 mm/s 10 mm/mV", noisy);
    gfx_text(c, 4, VIEW_RES_Y + 49, buf, &Font12, C_MUTED);
}

void ecg_view_init(void)
{
    draw_grid();
}

void ecg_view_status(const char *line1, const char *line2)
{
    gfx_clear(plat_layer(LAYER_SIGNAL), TRANSPARENT);
    gfx_canvas_t *c = plat_layer(LAYER_RESULT);
    gfx_clear(c, TRANSPARENT);
    draw_header(c, NULL);
    int y = VIEW_ECG_Y + VIEW_ECG_H / 2 - 20;
    gfx_fill_rect(c, 60, y - 8, LCD_W - 120, 52, ARGB(0xE0, 0xFF, 0xFF, 0xFF));
    text_center(c, LCD_W / 2, y, line1, &Font12, C_TEXT);
    text_center(c, LCD_W / 2, y + 16, line2, &Font12, C_MUTED);
    char buf[64];
    snprintf(buf, sizeof(buf), "%s  %lu.%lu M MACC", ECG_MODEL_TAG,
             (unsigned long)(ECG_MODEL_MACC / 1000000u), (unsigned long)(ECG_MODEL_MACC / 100000u % 10u));
    text_center(c, LCD_W / 2, y + 30, buf, &Font8, C_MUTED);
    plat_lcd_present();
}

void ecg_view_render(const int16_t *win, const ecg_result_t *r)
{
    draw_signal(win);
    draw_result(r);
    plat_lcd_present();
}
