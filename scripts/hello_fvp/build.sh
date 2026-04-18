#!/bin/bash
# 编译 hello.elf（Cortex-M55 裸机）
# 脚本内部自行配置 PATH，不依赖 .bashrc
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 环境自配置：强制使用 arm-gnu-14（支持 Cortex-M55） ---
export PATH="$HOME/arm-gnu-14/bin:$PATH"

GCC="$HOME/arm-gnu-14/bin/arm-none-eabi-gcc"
if [ ! -x "$GCC" ]; then
    echo "找不到 arm-gnu-14 工具链: $GCC"
    exit 1
fi

echo "=== Compiling for Cortex-M55 ==="
"$GCC" \
    -mcpu=cortex-m55 -mthumb \
    -O2 -g \
    -ffreestanding -nostdlib \
    -Wall -Wextra \
    -T corstone300.ld \
    -Wl,--gc-sections,-Map=hello.map \
    startup.c hello.c \
    -o hello.elf

echo "=== Build OK ==="
"$HOME/arm-gnu-14/bin/arm-none-eabi-size" hello.elf
echo ""
echo "ELF entry address:"
"$HOME/arm-gnu-14/bin/arm-none-eabi-readelf" -h hello.elf | grep "Entry"
