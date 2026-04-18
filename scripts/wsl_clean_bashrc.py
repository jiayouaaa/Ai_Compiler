#!/usr/bin/env python3
"""清理并修复 ~/.bashrc 的 Ethos-U/FVP 环境变量块。"""
import os

home = os.path.expanduser("~")
bashrc = os.path.join(home, ".bashrc")

with open(bashrc, "r") as f:
    lines = f.readlines()

# 过滤掉所有相关的旧行
keep = []
for line in lines:
    low = line.lower()
    if ("arm-gnu-14" in low or
        "fvp/corstone" in low or
        "ld_library_path" in low or
        "ethos-u / fvp" in line.lower()):
        continue
    keep.append(line)

# 去掉末尾的空白行
while keep and keep[-1].strip() == "":
    keep.pop()

# 追加干净的环境变量块（literal $HOME / $PATH，不做展开）
keep.append("\n")
keep.append("# === Ethos-U / FVP toolchain ===\n")
keep.append('export PATH="$HOME/arm-gnu-14/bin:$PATH"\n')
keep.append('export PATH="$HOME/fvp/corstone-300/models/Linux64_GCC-9.3:$PATH"\n')
keep.append('export LD_LIBRARY_PATH="$HOME/fvp/corstone-300/python/lib:$LD_LIBRARY_PATH"\n')

with open(bashrc, "w") as f:
    f.writelines(keep)

print("done. Tail of .bashrc:")
for line in keep[-6:]:
    print(line.rstrip())
