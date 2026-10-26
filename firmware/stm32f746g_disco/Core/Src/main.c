/*
 * STM32F746G-Discovery: ECG rhythm kit. Board layer (clocks, MPU / caches, USART1 + DMA,
 * LCD + DMA2D) and the main loop; the application itself is App/ (shared with the PC's
 * virtual kit, host/host_main.c).
 *
 * SDRAM (8 MB at 0xC0000000):
 *   lower 4 MB  non-cacheable (MPU): the LCD layers, read by DMA2D / LTDC
 *   upper 4 MB  cacheable write-back: the two networks' activations
 */
#include <string.h>

#include "ecg_app.h"
#include "model_meta.h"
#include "platform.h"
#include "stm32746g_discovery.h"
#include "stm32746g_discovery_lcd.h"
#include "stm32f7xx_hal.h"

#define FB_GRID    0xC0000000u      /* LTDC layer 0 */
#define FB_OUT0    0xC0080000u      /* LTDC layer 1, two buffers */
#define FB_OUT1    0xC0100000u
#define FB_SIGNAL  0xC0180000u
#define FB_RESULT  0xC0200000u
#define ACT_A      0xC0400000u
#define ACT_B      (ACT_A + ((ECG_MODEL_ACT_A + 0xFFFu) & ~0xFFFu))
#define FB_BYTES   (LCD_W * LCD_H * 4u)

_Static_assert(FB_RESULT + FB_BYTES <= ACT_A, "LCD layers overlap the activations");
_Static_assert(ACT_B + ECG_MODEL_ACT_B <= 0xC0800000u, "activations do not fit the SDRAM");

#define RX_SIZE 32768u

UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_rx;
static DMA2D_HandleTypeDef hdma2d;
static uint8_t rx_ring[RX_SIZE] __attribute__((section(".dtcm_bss"), aligned(32)));
static uint32_t rx_tail;
static uint32_t fb_front = FB_OUT0;
static gfx_canvas_t canvas[3] = {
    {(uint32_t *)FB_GRID, LCD_W, LCD_H},
    {(uint32_t *)FB_SIGNAL, LCD_W, LCD_H},
    {(uint32_t *)FB_RESULT, LCD_W, LCD_H},
};

void Error_Handler(void)
{
    __disable_irq();
    BSP_LED_Init(LED1);
    for (;;) {
        BSP_LED_Toggle(LED1);
        for (volatile uint32_t i = 0; i < 2000000u; i++)
            ;
    }
}

/* --- clocks, MPU, caches ----------------------------------------------------------------- */

static void clock_config(void)
{
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    osc.OscillatorType = RCC_OSCILLATORTYPE_HSE;           /* 25 MHz */
    osc.HSEState = RCC_HSE_ON;
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    osc.PLL.PLLM = 25;
    osc.PLL.PLLN = 432;
    osc.PLL.PLLP = RCC_PLLP_DIV2;                          /* 216 MHz */
    osc.PLL.PLLQ = 9;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK)
        Error_Handler();
    if (HAL_PWREx_EnableOverDrive() != HAL_OK)
        Error_Handler();
    clk.ClockType = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider = RCC_SYSCLK_DIV1;
    clk.APB1CLKDivider = RCC_HCLK_DIV4;
    clk.APB2CLKDivider = RCC_HCLK_DIV2;                    /* USART1: 108 MHz */
    if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_7) != HAL_OK)
        Error_Handler();
}

static void mpu_config(void)
{
    MPU_Region_InitTypeDef r = {0};
    HAL_MPU_Disable();
    r.Enable = MPU_REGION_ENABLE;
    r.AccessPermission = MPU_REGION_FULL_ACCESS;
    r.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
    r.IsShareable = MPU_ACCESS_NOT_SHAREABLE;
    r.TypeExtField = MPU_TEX_LEVEL1;
    r.SubRegionDisable = 0;
    /* all SDRAM: normal memory, write-back write-allocate */
    r.Number = MPU_REGION_NUMBER0;
    r.BaseAddress = 0xC0000000u;
    r.Size = MPU_REGION_SIZE_8MB;
    r.IsCacheable = MPU_ACCESS_CACHEABLE;
    r.IsBufferable = MPU_ACCESS_BUFFERABLE;
    HAL_MPU_ConfigRegion(&r);
    /* LCD layers: normal, non-cacheable (higher region number wins) */
    r.Number = MPU_REGION_NUMBER1;
    r.Size = MPU_REGION_SIZE_4MB;
    r.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
    r.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
    HAL_MPU_ConfigRegion(&r);
    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55u;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/* --- platform.h -------------------------------------------------------------------------- */

uint32_t plat_micros(void)
{
    static uint32_t last, high;          /* extend CYCCNT (wraps every ~19.9 s) */
    uint32_t now = DWT->CYCCNT;
    if (now < last)
        high++;
    last = now;
    return (uint32_t)((((uint64_t)high << 32) | now) / (SystemCoreClock / 1000000u));
}

uint32_t plat_cpu_hz(void) { return SystemCoreClock; }

void plat_uart_write(const uint8_t *data, size_t n)
{
    while (n) {
        uint16_t k = n > 0xFFFFu ? 0xFFFFu : (uint16_t)n;
        HAL_UART_Transmit(&huart1, (uint8_t *)data, k, 2000);
        data += k;
        n -= k;
    }
}

gfx_canvas_t *plat_layer(int which) { return &canvas[which]; }

void plat_fill(gfx_canvas_t *c, int x, int y, int w, int h, uint32_t argb)
{
    hdma2d.Instance = DMA2D;
    hdma2d.Init.Mode = DMA2D_R2M;
    hdma2d.Init.ColorMode = DMA2D_OUTPUT_ARGB8888;
    hdma2d.Init.OutputOffset = (uint32_t)(c->w - w);
    if (HAL_DMA2D_Init(&hdma2d) != HAL_OK)
        return;
    if (HAL_DMA2D_Start(&hdma2d, argb, (uint32_t)(c->px + y * c->w + x), (uint32_t)w, (uint32_t)h) == HAL_OK)
        HAL_DMA2D_PollForTransfer(&hdma2d, 100);
}

void plat_lcd_present(void)
{
    uint32_t back = fb_front == FB_OUT0 ? FB_OUT1 : FB_OUT0;
    hdma2d.Instance = DMA2D;
    hdma2d.Init.Mode = DMA2D_M2M_BLEND;
    hdma2d.Init.ColorMode = DMA2D_OUTPUT_ARGB8888;
    hdma2d.Init.OutputOffset = 0;
    for (int l = 0; l < 2; l++) {        /* 0 background = signal, 1 foreground = result */
        hdma2d.LayerCfg[l].InputColorMode = DMA2D_INPUT_ARGB8888;
        hdma2d.LayerCfg[l].AlphaMode = DMA2D_NO_MODIF_ALPHA;
        hdma2d.LayerCfg[l].InputAlpha = 0xFF;
        hdma2d.LayerCfg[l].InputOffset = 0;
    }
    if (HAL_DMA2D_Init(&hdma2d) != HAL_OK || HAL_DMA2D_ConfigLayer(&hdma2d, 0) != HAL_OK ||
        HAL_DMA2D_ConfigLayer(&hdma2d, 1) != HAL_OK)
        return;
    if (HAL_DMA2D_BlendingStart(&hdma2d, FB_RESULT, FB_SIGNAL, back, LCD_W, LCD_H) != HAL_OK)
        return;
    HAL_DMA2D_PollForTransfer(&hdma2d, 200);
    /* show it from the next vertical blanking on; wait for that so the old buffer is free */
    BSP_LCD_SetLayerAddress_NoReload(1, back);
    BSP_LCD_Reload(LCD_RELOAD_VERTICAL_BLANKING);
    uint32_t t0 = HAL_GetTick();
    while ((LTDC->SRCR & LTDC_SRCR_VBR) && HAL_GetTick() - t0 < 50)
        ;
    fb_front = back;
}

/* --- peripherals ------------------------------------------------------------------------- */

static void lcd_init(void)
{
    BSP_LCD_Init();                                  /* LTDC + SDRAM */
    BSP_LCD_LayerDefaultInit(0, FB_GRID);
    BSP_LCD_LayerDefaultInit(1, FB_OUT0);
    memset((void *)FB_OUT0, 0, FB_BYTES);            /* transparent until the first frame */
    BSP_LCD_SetLayerVisible(0, ENABLE);
    BSP_LCD_SetLayerVisible(1, ENABLE);
    BSP_LCD_DisplayOn();
}

static void uart_start_rx(void)
{
    rx_tail = 0;
    HAL_UART_Receive_DMA(&huart1, rx_ring, RX_SIZE);
}

static void uart_init(void)
{
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_DMA2_CLK_ENABLE();
    g.Mode = GPIO_MODE_AF_PP;
    g.Pull = GPIO_PULLUP;
    g.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    g.Alternate = GPIO_AF7_USART1;
    g.Pin = GPIO_PIN_9;                              /* PA9  TX -> ST-LINK VCP */
    HAL_GPIO_Init(GPIOA, &g);
    g.Pin = GPIO_PIN_7;                              /* PB7  RX <- ST-LINK VCP */
    HAL_GPIO_Init(GPIOB, &g);

    huart1.Instance = USART1;
    huart1.Init.BaudRate = APP_BAUD;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart1) != HAL_OK)
        Error_Handler();

    hdma_usart1_rx.Instance = DMA2_Stream2;          /* USART1_RX: DMA2 stream 2 channel 4 */
    hdma_usart1_rx.Init.Channel = DMA_CHANNEL_4;
    hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart1_rx.Init.Mode = DMA_CIRCULAR;
    hdma_usart1_rx.Init.Priority = DMA_PRIORITY_HIGH;
    hdma_usart1_rx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
    if (HAL_DMA_Init(&hdma_usart1_rx) != HAL_OK)
        Error_Handler();
    __HAL_LINKDMA(&huart1, hdmarx, hdma_usart1_rx);
    HAL_NVIC_SetPriority(DMA2_Stream2_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream2_IRQn);
    HAL_NVIC_SetPriority(USART1_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);
    uart_start_rx();
}

/* An overrun / framing error stops the HAL's DMA reception: start it again. */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        HAL_UART_AbortReceive(huart);
        uart_start_rx();
    }
}

/* --- main loop --------------------------------------------------------------------------- */

static void poll_rx(void)
{
    uint32_t head = RX_SIZE - __HAL_DMA_GET_COUNTER(huart1.hdmarx);
    if (head >= RX_SIZE)
        head = 0;
    if (head == rx_tail)
        return;
    if (head > rx_tail) {
        app_rx(rx_ring + rx_tail, head - rx_tail);
    } else {
        app_rx(rx_ring + rx_tail, RX_SIZE - rx_tail);
        app_rx(rx_ring, head);
    }
    rx_tail = head;
    BSP_LED_Toggle(LED1);
}

int main(void)
{
    mpu_config();
    SCB_EnableICache();
    SCB_EnableDCache();
    HAL_Init();
    clock_config();
    dwt_init();
    BSP_LED_Init(LED1);
    lcd_init();
    uart_init();
    __HAL_RCC_CRC_CLK_ENABLE();                      /* the X-CUBE-AI runtime checks for an STM32 CRC unit */
    if (app_init((void *)ACT_A, (void *)ACT_B) != 0)
        Error_Handler();
    for (;;)
        poll_rx();
}
