/*
 * Minimal ARGB8888 canvas: rectangles, lines, BSP bitmap fonts. Drawing writes pixels (no
 * blending inside a layer); layers are blended when they are put on the display.
 */
#ifndef GFX_H
#define GFX_H

#include <stdint.h>

#include "fonts.h"

#define LCD_W 480
#define LCD_H 272

#define ARGB(a, r, g, b) (((uint32_t)(a) << 24) | ((uint32_t)(r) << 16) | ((uint32_t)(g) << 8) | (uint32_t)(b))
#define RGB(rgb)         (0xFF000000u | (uint32_t)(rgb))
#define WITH_ALPHA(argb, a) (((argb) & 0x00FFFFFFu) | ((uint32_t)(a) << 24))
#define TRANSPARENT      0x00000000u

typedef struct {
    uint32_t *px;     /* row-major, pitch = w */
    int w, h;
} gfx_canvas_t;

void gfx_clear(gfx_canvas_t *c, uint32_t argb);
void gfx_fill_rect(gfx_canvas_t *c, int x, int y, int w, int h, uint32_t argb);
void gfx_hline(gfx_canvas_t *c, int x, int y, int w, uint32_t argb);
void gfx_vline(gfx_canvas_t *c, int x, int y, int h, uint32_t argb);
void gfx_dashed_vline(gfx_canvas_t *c, int x, int y, int h, int on, int off, uint32_t argb);
void gfx_line(gfx_canvas_t *c, int x0, int y0, int x1, int y1, uint32_t argb);
void gfx_text(gfx_canvas_t *c, int x, int y, const char *s, const sFONT *font, uint32_t argb);
int gfx_text_width(const char *s, const sFONT *font);

#endif
