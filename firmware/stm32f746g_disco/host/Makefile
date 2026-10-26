# Virtual kit: App/ + Model/ built for the PC (see host_main.c). Needs export_model.py to
# have run (it leaves the x86 X-CUBE-AI runtime in ../build/host_ai).
#   make -C firmware/stm32f746g_disco/host        -> ../build/host/ecgkit_sim

ROOT    := ..
CUBE    := $(ROOT)/third_party/STM32CubeF7
AI      := $(ROOT)/build/host_ai
OUT     := $(ROOT)/build/host
TARGET  := $(OUT)/ecgkit_sim

SRCS := $(wildcard $(ROOT)/App/*.c) host_main.c \
        $(wildcard $(ROOT)/Model/ecga*.c) $(wildcard $(ROOT)/Model/ecgb*.c) \
        $(CUBE)/Utilities/Fonts/font8.c $(CUBE)/Utilities/Fonts/font12.c
OBJS := $(patsubst %.c,$(OUT)/obj/%.o,$(notdir $(SRCS)))

CFLAGS  += -std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
           -I$(ROOT)/App -I$(ROOT)/Model -I$(AI)/include -I$(CUBE)/Utilities/Fonts
LDLIBS  += -L$(AI)/lib -lruntime -lst_cmsis_nn -lcmsis-nn -lm

vpath %.c $(sort $(dir $(SRCS)))

all: $(TARGET)

$(AI)/include:
	$(error $(AI) is missing: run python3 firmware/stm32f746g_disco/export_model.py first)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDLIBS) -o $@

$(OUT)/obj/%.o: %.c | $(AI)/include $(CUBE)/Utilities/Fonts
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -c $< -o $@

$(CUBE)/Utilities/Fonts:
	$(error STM32CubeF7 is missing: run firmware/stm32f746g_disco/fetch_deps.sh)

clean:
	rm -rf $(OUT)

-include $(OBJS:.o=.d)
.PHONY: all clean
