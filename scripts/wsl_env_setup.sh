#!/bin/bash
# WSL Ubuntu 环境变量设置脚本
# 在 WSL 内 source 此文件（或追加到 .bashrc）

export PATH="$HOME/arm-gnu-14/bin:$PATH"
export PATH="$HOME/fvp/corstone-300/models/Linux64_GCC-9.3:$PATH"
export LD_LIBRARY_PATH="$HOME/fvp/corstone-300/python/lib:$LD_LIBRARY_PATH"
