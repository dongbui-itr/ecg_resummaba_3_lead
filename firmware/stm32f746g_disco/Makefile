# STM32F746G-Discovery ECG rhythm kit.
#
#   ./fetch_deps.sh                    once: STM32CubeF7 (HAL, BSP, fonts) -> third_party/
#   python3 export_model.py            after a model change: Model/ (needs TF + X-CUBE-AI)
#   make                               -> build/ecg_kit.elf / .bin / .hex
#   make flash                         STM32_Programmer_CLI over the on-board ST-LINK
#   make APP_BAUD=115200               another UART speed (the web tab must use the same)

TARGET   := ecg_kit
BUILD    := build/fw
CUBE     := third_party/STM32CubeF7
-include Model/xcubeai.mk
XCUBEAI  ?= $(lastword $(wildcard $(HOME)/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.*))
AI_LIB   := $(firstword $(wildcard $(XCUBEAI)/Middlewares/ST/AI/Lib/GCC/ARMCortexM7/NetworkRuntime*_CM7_GCC.a))
APP_BAUD ?= 921600

PREFIX   ?= arm-none-eabi-
CC       := $(PREFIX)gcc
AS       := $(PREFIX)gcc -x assembler-with-cpp
OBJCOPY  := $(PREFIX)objcopy
SIZE     := $(PREFIX)size
PROGRAMMER ?= $(firstword $(wildcard $(HOME)/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI) STM32_Programmer_CLI)

HAL      := $(CUBE)/Drivers/STM32F7xx_HAL_Driver/Src
BSP      := $(CUBE)/Drivers/BSP/STM32746G-Discovery
HAL_SRCS := stm32f7xx_hal.c stm32f7xx_hal_cortex.c stm32f7xx_hal_rcc.c stm32f7xx_hal_rcc_ex.c \
            stm32f7xx_hal_pwr.c stm32f7xx_hal_pwr_ex.c stm32f7xx_hal_gpio.c stm32f7xx_hal_dma.c \
            stm32f7xx_hal_dma_ex.c stm32f7xx_hal_dma2d.c stm32f7xx_hal_ltdc.c stm32f7xx_hal_ltdc_ex.c \
            stm32f7xx_hal_sdram.c stm32f7xx_ll_fmc.c stm32f7xx_hal_uart.c stm32f7xx_hal_uart_ex.c \
            stm32f7xx_hal_i2c.c stm32f7xx_hal_i2c_ex.c stm32f7xx_hal_flash.c stm32f7xx_hal_flash_ex.c

SRCS := $(wildcard Core/Src/*.c) $(wildcard App/*.c) \
        $(wildcard Model/ecga*.c) $(wildcard Model/ecgb*.c) \
        $(addprefix $(HAL)/,$(HAL_SRCS)) \
        $(BSP)/stm32746g_discovery.c $(BSP)/stm32746g_discovery_lcd.c $(BSP)/stm32746g_discovery_sdram.c \
        $(CUBE)/Drivers/CMSIS/Device/ST/STM32F7xx/Source/Templates/system_stm32f7xx.c
ASRCS := $(CUBE)/Drivers/CMSIS/Device/ST/STM32F7xx/Source/Templates/gcc/startup_stm32f746xx.s
OBJS  := $(addprefix $(BUILD)/,$(notdir $(SRCS:.c=.o) $(ASRCS:.s=.o)))

MCU     := -mcpu=cortex-m7 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
DEFS    := -DSTM32F746xx -DUSE_HAL_DRIVER -DUSE_STM32746G_DISCO -DAPP_BAUD=$(APP_BAUD)
INCS    := -ICore/Inc -IApp -IModel -I$(XCUBEAI)/Middlewares/ST/AI/Inc \
           -I$(CUBE)/Drivers/STM32F7xx_HAL_Driver/Inc -I$(CUBE)/Drivers/CMSIS/Device/ST/STM32F7xx/Include \
           -I$(CUBE)/Drivers/CMSIS/Include -I$(BSP) -I$(CUBE)/Utilities/Fonts
CFLAGS  := $(MCU) $(DEFS) $(INCS) -std=gnu11 -O2 -g -Wall -ffunction-sections -fdata-sections \
           -fno-strict-aliasing
LDFLAGS := $(MCU) -T STM32F746NGHx_FLASH.ld -specs=nano.specs -specs=nosys.specs \
           -Wl,--gc-sections -Wl,-Map=$(BUILD)/$(TARGET).map -Wl,--print-memory-usage

vpath %.c $(sort $(dir $(SRCS)))
vpath %.s $(sort $(dir $(ASRCS)))

all: check $(BUILD)/$(TARGET).bin $(BUILD)/$(TARGET).hex

check:
	@test -d $(CUBE)/Drivers/STM32F7xx_HAL_Driver/Src || { echo "run ./fetch_deps.sh first"; exit 1; }
	@test -f Model/model_meta.h || { echo "run python3 export_model.py first"; exit 1; }
	@test -f "$(AI_LIB)" || { echo "X-CUBE-AI runtime not found under XCUBEAI=$(XCUBEAI)"; exit 1; }

$(BUILD)/$(TARGET).elf: $(OBJS)
	$(CC) $(OBJS) $(AI_LIB) $(LDFLAGS) -lm -o $@
	$(SIZE) $@

$(BUILD)/%.bin: $(BUILD)/%.elf
	$(OBJCOPY) -O binary $< $@

$(BUILD)/%.hex: $(BUILD)/%.elf
	$(OBJCOPY) -O ihex $< $@

$(BUILD)/%.o: %.c | $(BUILD)
	$(CC) $(CFLAGS) -MMD -c $< -o $@

$(BUILD)/%.o: %.s | $(BUILD)
	$(AS) $(MCU) -c $< -o $@

$(BUILD):
	mkdir -p $@

flash: all
	$(PROGRAMMER) -c port=SWD -w $(BUILD)/$(TARGET).hex -v -rst

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)
.PHONY: all check flash clean
