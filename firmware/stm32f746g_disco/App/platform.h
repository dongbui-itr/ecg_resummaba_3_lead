/*
 * What the portable application (App/) needs from the board. Implemented twice:
 * Core/Src/board.c (STM32F746G-Discovery: HAL, DMA2D, LTDC) and host/host_main.c (the PC's
 * virtual kit: stdin/stdout or a pty, software blending, PPM snapshots).
 */
#ifndef PLATFORM_H
#define PLATFORM_H

#include <stddef.h>
#include <stdint.h>

#include "gfx.h"

/* Free-running microsecond clock; only differences < ~19 s are meaningful. */
uint32_t plat_micros(void);

/* Blocking write of `n` bytes to the PC link. */
void plat_uart_write(const uint8_t *data, size_t n);

/* The three LCD layers: grid (opaque, drawn once), signal and result (ARGB, transparent
 * background), each LCD_W x LCD_H. */
gfx_canvas_t *plat_layer(int which);
enum { LAYER_GRID = 0, LAYER_SIGNAL = 1, LAYER_RESULT = 2 };

/* Fill w x h pixels of `c` at (x, y) with one ARGB value (DMA2D on the board). */
void plat_fill(gfx_canvas_t *c, int x, int y, int w, int h, uint32_t argb);

/* Put result-over-signal on the display (the grid is the layer underneath). */
void plat_lcd_present(void);

/* Board / build description for the INFO message. */
uint32_t plat_cpu_hz(void);

#endif
