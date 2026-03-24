cmake_minimum_required(VERSION 3.20)

project(self_compiler LANGUAGES CXX)

option(SELF_COMPILER_ENABLE_MLIR "启用 MLIR dialect/lowering 集成" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(self_compiler_core
    src/ir/op_registry.cpp
    src/frontend/transformer_block_builder.cpp
    src/frontend/importer_factory.cpp
    src/frontend/json_importer.cpp
    src/frontend/llama_config_importer.cpp
    src/frontend/onnx_importer.cpp
    src/passes/pass_manager.cpp
    src/passes/canonicalize_pass.cpp
    src/passes/recognize_onnx_ops_pass.cpp
    src/passes/graph_partition_pass.cpp
    src/passes/lower_transformer_to_runtime_pass.cpp
    src/memory/live_interval.cpp
    src/memory/memory_planner.cpp
    src/backend/command_stream.cpp
    src/backend/toy_backend.cpp
    src/backend/ethosu_backend.cpp
    src/app/compiler_app.cpp
    src/mlir/mlir_bridge_stub.cpp
)

target_include_directories(self_compiler_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(self_compiler_core
    PUBLIC
        $<$<BOOL:${SELF_COMPILER_ENABLE_MLIR}>:SELF_COMPILER_ENABLE_MLIR=1>
)

if(SELF_COMPILER_ENABLE_MLIR)
    find_package(MLIR CONFIG QUIET)
    if(NOT MLIR_FOUND)
        message(FATAL_ERROR "已开启 SELF_COMPILER_ENABLE_MLIR，但当前未找到 MLIR。请先设置 MLIR_DIR / LLVM_DIR。")
    endif()
    target_compile_definitions(self_compiler_core PUBLIC SELF_COMPILER_HAS_MLIR=1)
endif()

add_executable(self_compiler_cli tools/self_compiler_main.cpp)
target_link_libraries(self_compiler_cli PRIVATE self_compiler_core)

enable_testing()
add_test(NAME self_compiler_smoke COMMAND self_compiler_cli --mode demo)
