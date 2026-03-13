param(
    [ValidateSet("cli", "ctest", "both")]
    [string]$Mode = "cli",
    [string]$BuildDir = "build/core-debug-mingw",
    [string]$InputPath = "",
    [string]$Format = "auto",
    [string[]]$ExtraCliArgs = @(),
    [string]$LogDir = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$resolvedBuildDir = Join-Path $projectRoot $BuildDir

if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = Join-Path $resolvedBuildDir "utf8-logs"
} elseif (-not [System.IO.Path]::IsPathRooted($LogDir)) {
    $LogDir = Join-Path $projectRoot $LogDir
}

function Enable-Utf8Console {
    chcp.com 65001 | Out-Null

    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $global:OutputEncoding = $utf8NoBom
    $PSDefaultParameterValues["Out-File:Encoding"] = "utf8"
    $PSDefaultParameterValues["Set-Content:Encoding"] = "utf8"
}

function Invoke-Utf8Process {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)]
        [string]$LogName
    )

    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir | Out-Null
    }

    $logPath = Join-Path $LogDir $LogName
    $commandText = $FilePath
    if ($Arguments.Count -gt 0) {
        $commandText += " " + ($Arguments -join " ")
    }

    Write-Host ("[utf8-run] " + $commandText)

    $output = & $FilePath @Arguments 2>&1
    $exitCode = $LASTEXITCODE

    $output | Out-File -FilePath $logPath -Encoding utf8
    foreach ($line in $output) {
        Write-Host $line
    }

    Write-Host ("[utf8-log] " + $logPath)

    if ($exitCode -ne 0) {
        throw ("Command failed with exit code " + $exitCode + ": " + $commandText)
    }
}

Enable-Utf8Console

if ($Mode -eq "ctest" -or $Mode -eq "both") {
    Invoke-Utf8Process -FilePath "ctest" `
        -Arguments @("--test-dir", $resolvedBuildDir, "--output-on-failure") `
        -LogName "ctest.log"
}

if ($Mode -eq "cli" -or $Mode -eq "both") {
    $cliPath = Join-Path $resolvedBuildDir "self_compiler_cli.exe"
    if (-not (Test-Path $cliPath)) {
        throw ("CLI not found: " + $cliPath)
    }

    $cliArgs = @("--mode", "demo")
    if (-not [string]::IsNullOrWhiteSpace($InputPath)) {
        $cliArgs += @("--input", $InputPath)
    }
    if (-not [string]::IsNullOrWhiteSpace($Format) -and $Format -ne "auto") {
        $cliArgs += @("--format", $Format)
    }
    if ($ExtraCliArgs.Count -gt 0) {
        $cliArgs += $ExtraCliArgs
    }

    Invoke-Utf8Process -FilePath $cliPath `
        -Arguments $cliArgs `
        -LogName "self_compiler_cli.log"
}
