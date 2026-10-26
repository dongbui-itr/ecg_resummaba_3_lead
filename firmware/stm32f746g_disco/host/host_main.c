/*
 * Virtual kit: the firmware's portable code (App/) and the generated networks (Model/),
 * built for the PC against the x86 X-CUBE-AI runtime, speaking the same protocol.
 *
 *   ecgkit_sim                       frames on stdin / stdout
 *   ecgkit_sim --pty                 serve on a new pseudo-terminal; its path is printed
 *                                    first ("PTY /dev/pts/N") - open it like the board's COM port
 *   ecgkit_sim ... --snap lcd.ppm    after every LCD update, write the 480x272 display (the
 *                                    three layers blended as LTDC + DMA2D do) to that file
 *
 * Only the board layer differs from the firmware: no HAL, no DMA2D/LTDC (software blending),
 * timing from the PC clock.
 */
#define _DEFAULT_SOURCE                 /* cfmakeraw */
#define _XOPEN_SOURCE 600               /* posix_openpt, ptsname */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "ecg_app.h"
#include "model_meta.h"
#include "platform.h"

static int out_fd = 1;
static const char *snap_path;
static uint32_t layers[3][LCD_W * LCD_H];
static gfx_canvas_t canvas[3] = {
    {layers[0], LCD_W, LCD_H}, {layers[1], LCD_W, LCD_H}, {layers[2], LCD_W, LCD_H}};

uint32_t plat_micros(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000000ull + ts.tv_nsec / 1000);
}

uint32_t plat_cpu_hz(void) { return 0; }       /* 0 = virtual kit */

void plat_uart_write(const uint8_t *data, size_t n)
{
    while (n) {
        ssize_t k = write(out_fd, data, n);
        if (k < 0) {
            if (errno == EINTR || errno == EAGAIN)
                continue;
            return;
        }
        data += k;
        n -= (size_t)k;
    }
}

gfx_canvas_t *plat_layer(int which) { return &canvas[which]; }

void plat_fill(gfx_canvas_t *c, int x, int y, int w, int h, uint32_t argb)
{
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
            c->px[(y + j) * c->w + x + i] = argb;
}

/* fg over bg, straight alpha (DMA2D blending), then over the opaque grid (LTDC layer 1 on
 * layer 0 with constant alpha 255 x pixel alpha) */
static void compose(uint8_t *rgb)
{
    for (int i = 0; i < LCD_W * LCD_H; i++) {
        uint32_t f = layers[LAYER_RESULT][i], b = layers[LAYER_SIGNAL][i], g = layers[LAYER_GRID][i];
        float af = (f >> 24) / 255.0f, ab = (b >> 24) / 255.0f;
        float ao = af + ab * (1.0f - af);
        for (int ch = 0; ch < 3; ch++) {
            int sh = 16 - 8 * ch;
            float cf = (f >> sh) & 0xFF, cb = (b >> sh) & 0xFF, cg = (g >> sh) & 0xFF;
            float c1 = ao > 0.0f ? (cf * af + cb * ab * (1.0f - af)) / ao : 0.0f;
            float c = c1 * ao + cg * (1.0f - ao);
            rgb[i * 3 + ch] = (uint8_t)(c + 0.5f);
        }
    }
}

void plat_lcd_present(void)
{
    if (!snap_path)
        return;
    static uint8_t rgb[LCD_W * LCD_H * 3];
    compose(rgb);
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp", snap_path);
    FILE *f = fopen(tmp, "wb");
    if (!f)
        return;
    fprintf(f, "P6\n%d %d\n255\n", LCD_W, LCD_H);
    fwrite(rgb, 1, sizeof(rgb), f);
    fclose(f);
    rename(tmp, snap_path);
}

static int open_pty(void)
{
    int fd = posix_openpt(O_RDWR | O_NOCTTY);
    if (fd < 0 || grantpt(fd) || unlockpt(fd))
        return -1;
    struct termios t;
    if (tcgetattr(fd, &t) == 0) {
        cfmakeraw(&t);
        tcsetattr(fd, TCSANOW, &t);
    }
    printf("PTY %s\n", ptsname(fd));
    fflush(stdout);
    return fd;
}

int main(int argc, char **argv)
{
    int in_fd = 0, pty = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pty"))
            pty = 1;
        else if (!strcmp(argv[i], "--snap") && i + 1 < argc)
            snap_path = argv[++i];
        else {
            fprintf(stderr, "usage: %s [--pty] [--snap file.ppm]\n", argv[0]);
            return 2;
        }
    }
    if (pty) {
        in_fd = out_fd = open_pty();
        if (in_fd < 0) {
            perror("pty");
            return 1;
        }
    }
    void *act_a = aligned_alloc(32, (ECG_MODEL_ACT_A + 31) / 32 * 32);
    void *act_b = aligned_alloc(32, (ECG_MODEL_ACT_B + 31) / 32 * 32);
    if (app_init(act_a, act_b) != 0) {
        fprintf(stderr, "model init failed\n");
        return 1;
    }
    uint8_t buf[4096];
    for (;;) {
        ssize_t n = read(in_fd, buf, sizeof(buf));
        if (n > 0) {
            app_rx(buf, (uint32_t)n);
        } else if (n == 0 && !pty) {
            break;                                  /* stdin closed */
        } else if (n < 0 && errno != EINTR) {
            if (!pty)
                break;
            usleep(20000);                          /* pty: no client attached yet */
        }
    }
    return 0;
}
