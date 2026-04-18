#!/bin/bash
# 在 Corstone-300 FVP 上运行 hello.elf
# 脚本内部自行配置 PATH / LD_LIBRARY_PATH，不依赖 .bashrc
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 环境自配置：确保能找到 FVP 二进制和它依赖的 Python 库 ---
export PATH="$HOME/fvp/corstone-300/models/Linux64_GCC-9.3:$PATH"
export LD_LIBRARY_PATH="$HOME/fvp/corstone-300/python/lib:$LD_LIBRARY_PATH"

if [ ! -f hello.elf ]; then
    echo "hello.elf 不存在，先运行 ./build.sh"
    exit 1
fi

FVP_BIN="$HOME/fvp/corstone-300/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55"
if [ ! -x "$FVP_BIN" ]; then
    echo "找不到 FVP: $FVP_BIN"
    exit 1
fi

echo "=== Launching FVP Corstone-300 ==="
"$FVP_BIN" \
    -a cpu0=hello.elf \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0 \
    -C mps3_board.uart0.out_file='-' \
    -C mps3_board.uart0.unbuffered_output=1 \
    -C cpu0.semihosting-enable=1 \
    --simlimit 5
