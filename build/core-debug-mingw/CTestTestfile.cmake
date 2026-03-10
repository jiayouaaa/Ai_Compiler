# CMake generated Testfile for 
# Source directory: D:/Ai_kernel/llama3/self_compiler
# Build directory: D:/Ai_kernel/llama3/self_compiler/build/core-debug-mingw
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[self_compiler_smoke]=] "D:/Ai_kernel/llama3/self_compiler/build/core-debug-mingw/self_compiler_cli.exe" "--mode" "demo")
set_tests_properties([=[self_compiler_smoke]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/Ai_kernel/llama3/self_compiler/CMakeLists.txt;49;add_test;D:/Ai_kernel/llama3/self_compiler/CMakeLists.txt;0;")
