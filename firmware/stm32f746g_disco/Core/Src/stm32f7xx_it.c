#include "stm32f7xx_hal.h"

extern UART_HandleTypeDef huart1;
extern DMA_HandleTypeDef hdma_usart1_rx;
void Error_Handler(void);

void NMI_Handler(void) {}
void HardFault_Handler(void) { Error_Handler(); }
void MemManage_Handler(void) { Error_Handler(); }
void BusFault_Handler(void) { Error_Handler(); }
void UsageFault_Handler(void) { Error_Handler(); }
void SVC_Handler(void) {}
void DebugMon_Handler(void) {}
void PendSV_Handler(void) {}
void SysTick_Handler(void) { HAL_IncTick(); }

void DMA2_Stream2_IRQHandler(void) { HAL_DMA_IRQHandler(&hdma_usart1_rx); }
void USART1_IRQHandler(void) { HAL_UART_IRQHandler(&huart1); }
