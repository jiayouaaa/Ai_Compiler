$ErrorActionPreference = "Continue"

Write-Host "[self_compiler] checking development environment..."

$tools = @("cmake", "g++", "mingw32-make", "ninja", "llvm-config")
foreach ($tool in $tools) {
    $found = Get-Command $tool -ErrorAction SilentlyContinue
    if ($null -eq $found) {
        Write-Host ("[missing] " + $tool)
    } else {
        Write-Host ("[ok] " + $tool + " -> " + $found.Source)
    }
}

Write-Host ""
Write-Host "[self_compiler] current MLIR env"
Write-Host ("LLVM_DIR=" + $env:LLVM_DIR)
Write-Host ("MLIR_DIR=" + $env:MLIR_DIR)

if (-not (Test-Path .\build)) {
    New-Item -ItemType Directory -Path .\build | Out-Null
    Write-Host "[ok] created build directory"
}

Write-Host ""
Write-Host "Core build commands:"
Write-Host "  cmake --preset core-debug"
Write-Host "  cmake --build --preset build-core-debug"
Write-Host ""
Write-Host "MLIR build commands (after LLVM/MLIR installation):"
Write-Host "  cmake --preset mlir-debug"
Write-Host "  cmake --build --preset build-mlir-debug"
