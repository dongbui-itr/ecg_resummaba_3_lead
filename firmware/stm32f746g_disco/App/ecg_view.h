/*
 * The 480 x 272 LCD in three layers (the web tab draws the same thing, webservice/static/
 * index.html 'kit' section - keep the two in step):
 *
 *   1 grid    ECG paper at 25 mm/s, 10 mm/mV (1 mm = 1.92 px, minor 1 mm / major 5 mm),
 *             header and result-panel backgrounds - drawn once, LTDC layer 0
 *   2 signal  the three leads of the current 10 s window, lead names
 *   3 result  header text, non-sinus bands, overlap marker, noise, beats (strip above the
 *             leads: AI boxes, reference letters under them), per-second REF / AI rows,
 *             summary lines
 *
 * Layers 2 and 3 are ARGB with a transparent background; 3 is blended over 2 into LTDC
 * layer 1 (DMA2D), which the LTDC blends over layer 0.
 */
#ifndef ECG_VIEW_H
#define ECG_VIEW_H

#include <stdint.h>

#include "ecg_app.h"

#define VIEW_HDR_H     20
#define VIEW_ECG_Y     20
#define VIEW_ECG_H     192          /* beat strip + 3 lanes, all on ECG paper */
#define VIEW_BEAT_H    22           /* AI beat boxes (top) and reference beat letters */
#define VIEW_LANE_Y    (VIEW_ECG_Y + VIEW_BEAT_H)
#define VIEW_LANE_H    56
#define VIEW_RES_Y     212
#define VIEW_PX_PER_S  48           /* 480 px / 10 s */
#define VIEW_PX_PER_MV 19.2f        /* 10 mm/mV at 1.92 px/mm */

void ecg_view_init(void);
/* Idle / status screen: two centred lines over the empty grid. */
void ecg_view_status(const char *line1, const char *line2);
/* One window: win (2500, 3) microvolts and its result. */
void ecg_view_render(const int16_t *win, const ecg_result_t *r);

#endif
