
Last updated: 2026-03-23

Quick recovery:
- Branch: `self-do`
- Main WIP: stage-1 ONNX op recognition pass has been added but not committed yet.
- Relevant files:
  - `include/self_compiler/passes/recognize_onnx_ops_pass.h`
  - `src/passes/recognize_onnx_ops_pass.cpp`
  - `src/app/compiler_app.cpp`
  - `CMakeLists.txt`

Current pass order:
- `CanonicalizePass`
- `RecognizeOnnxOpsPass`
- `LowerTransformerToRuntimePass`
- `CanonicalizePass`

Important:
- `build/core-debug-mingw` is currently in a bad/locked CMake state.
- If `cmake --build --preset build-core-debug` fails on `cmake.check_cache`, prefer a fresh build dir first.

Persistent memory:
- `C:\Users\16929\.codex\memories\self_compiler_memory.md`
