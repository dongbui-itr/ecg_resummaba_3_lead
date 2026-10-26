#!/usr/bin/env bash
# Fetch the parts of STM32CubeF7 this firmware builds against into third_party/STM32CubeF7
# (sparse clone: CMSIS, the F7 HAL, the STM32746G-Discovery BSP + its LCD components and the
# BSP fonts - ~40 MB instead of the full ~1 GB package). Run once; `make` refuses to start
# without it.
set -euo pipefail
cd "$(dirname "$0")"
DST=third_party/STM32CubeF7
TAG=v1.17.4
if [ -d "$DST/Drivers/STM32F7xx_HAL_Driver/Src" ]; then
    echo "already there: $DST"; exit 0
fi
mkdir -p third_party
if [ ! -d "$DST/.git" ]; then
    git -c advice.detachedHead=false clone --depth 1 --branch "$TAG" --filter=blob:none --sparse \
        https://github.com/STMicroelectronics/STM32CubeF7.git "$DST"
fi
cd "$DST"
# --skip-checks: the HAL / device / BSP folders are submodules (gitlinks), not directories
git sparse-checkout set --skip-checks Drivers/CMSIS/Include Drivers/CMSIS/Device/ST/STM32F7xx \
    Drivers/STM32F7xx_HAL_Driver Drivers/BSP/STM32746G-Discovery Drivers/BSP/Components/Common \
    Drivers/BSP/Components/rk043fn48h Drivers/BSP/Components/ft5336 Utilities/Fonts
for sm in Drivers/STM32F7xx_HAL_Driver Drivers/CMSIS/Device/ST/STM32F7xx \
          Drivers/BSP/STM32746G-Discovery Drivers/BSP/Components/Common \
          Drivers/BSP/Components/rk043fn48h Drivers/BSP/Components/ft5336; do
    git submodule update --init --depth 1 "$sm"
done
echo "done: $(pwd)"
