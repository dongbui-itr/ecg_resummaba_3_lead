#include "gfx.h"

#include "platform.h"

static int clip(int *x, int *y, int *w, int *h, const gfx_canvas_t *c)
{
    if (*x < 0) { *w += *x; *x = 0; }
    if (*y < 0) { *h += *y; *y = 0; }
    if (*x + *w > c->w) *w = c->w - *x;
    if (*y + *h > c->h) *h = c->h - *y;
    return *w > 0 && *h > 0;
}

void gfx_clear(gfx_canvas_t *c, uint32_t argb)
{
    plat_fill(c, 0, 0, c->w, c->h, argb);
}

void gfx_fill_rect(gfx_canvas_t *c, int x, int y, int w, int h, uint32_t argb)
{
    if (!clip(&x, &y, &w, &h, c))
        return;
    if (w * h >= 256) {               /* worth a DMA2D transfer */
        plat_fill(c, x, y, w, h, argb);
        return;
    }
    for (int j = 0; j < h; j++) {
        uint32_t *row = c->px + (y + j) * c->w + x;
        for (int i = 0; i < w; i++)
            row[i] = argb;
    }
}

void gfx_hline(gfx_canvas_t *c, int x, int y, int w, uint32_t argb)
{
    gfx_fill_rect(c, x, y, w, 1, argb);
}

void gfx_vline(gfx_canvas_t *c, int x, int y, int h, uint32_t argb)
{
    gfx_fill_rect(c, x, y, 1, h, argb);
}

void gfx_dashed_vline(gfx_canvas_t *c, int x, int y, int h, int on, int off, uint32_t argb)
{
    for (int j = 0; j < h; j += on + off)
        gfx_vline(c, x, y + j, (j + on > h) ? h - j : on, argb);
}

static inline void put(gfx_canvas_t *c, int x, int y, uint32_t argb)
{
    if ((unsigned)x < (unsigned)c->w && (unsigned)y < (unsigned)c->h)
        c->px[y * c->w + x] = argb;
}

void gfx_line(gfx_canvas_t *c, int x0, int y0, int x1, int y1, uint32_t argb)
{
    int dx = x1 > x0 ? x1 - x0 : x0 - x1, sx = x0 < x1 ? 1 : -1;
    int dy = y1 > y0 ? y0 - y1 : y1 - y0, sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        put(c, x0, y0, argb);
        if (x0 == x1 && y0 == y1)
            break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

/* BSP font tables: Height rows per glyph, (Width + 7) / 8 bytes per row, MSB = left pixel,
 * glyphs from ' ' to '~' (same decoding as BSP_LCD_DrawChar). */
static void draw_char(gfx_canvas_t *c, int x, int y, char ch, const sFONT *f, uint32_t argb)
{
    if (ch < ' ' || ch > '~')
        ch = '?';
    int bpr = (f->Width + 7) / 8;
    int pad = 8 * bpr - f->Width;
    const uint8_t *g = &f->table[(ch - ' ') * f->Height * bpr];
    for (int r = 0; r < f->Height; r++, g += bpr) {
        uint32_t line = g[0];
        if (bpr > 1) line = (line << 8) | g[1];
        if (bpr > 2) line = (line << 8) | g[2];
        for (int k = 0; k < f->Width; k++)
            if (line & (1u << (f->Width - k + pad - 1)))
                put(c, x + k, y + r, argb);
    }
}

void gfx_text(gfx_canvas_t *c, int x, int y, const char *s, const sFONT *font, uint32_t argb)
{
    for (; *s; s++, x += font->Width)
        draw_char(c, x, y, *s, font, argb);
}

int gfx_text_width(const char *s, const sFONT *font)
{
    int n = 0;
    while (s[n])
        n++;
    return n * font->Width;
}
