# build-daemon.ps1 — Build the crumbs-daemon sidecar for Windows and copy it
# to the path that Tauri's `externalBin` resolver expects.
#
# Usage:
#   .\scripts\build-daemon.ps1              # native release build
#   .\scripts\build-daemon.ps1 -Debug       # debug build
#   .\scripts\build-daemon.ps1 -Target x86_64-pc-windows-msvc
#
# Tauri's externalBin convention:
#   src-tauri\binaries\crumbs-daemon-<target-triple>.exe
#   e.g. src-tauri\binaries\crumbs-daemon-x86_64-pc-windows-msvc.exe

[CmdletBinding()]
param(
    [switch]$Debug,
    [string]$Target = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir    = Split-Path -Parent $ScriptDir

# ---------------------------------------------------------------------------
# Build mode
# ---------------------------------------------------------------------------
$BuildMode   = if ($Debug) { "debug" } else { "release" }
$CargoFlags  = if ($Debug) { @() } else { @("--release") }

# ---------------------------------------------------------------------------
# Determine target triple
# ---------------------------------------------------------------------------
if ($Target -eq "") {
    $rustcInfo = & rustc -vV 2>&1
    $Target    = ($rustcInfo | Select-String "^host: ").Line -replace "^host: ", ""
}

Write-Host "==> Building crumbs-daemon (mode=$BuildMode, target=$Target)"

# ---------------------------------------------------------------------------
# Cargo build
# ---------------------------------------------------------------------------
Push-Location $RootDir
try {
    & cargo build -p crumbs-daemon @CargoFlags --target $Target
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

$SrcBin = Join-Path $RootDir "target\$Target\$BuildMode\crumbs-daemon.exe"
$DstDir = Join-Path $RootDir "src-tauri\binaries"
$DstBin = Join-Path $DstDir  "crumbs-daemon-$Target.exe"

if (-not (Test-Path $DstDir)) {
    New-Item -ItemType Directory -Path $DstDir | Out-Null
}

Write-Host "==> Copying binary"
Write-Host "    $SrcBin"
Write-Host "    -> $DstBin"
Copy-Item -Path $SrcBin -Destination $DstBin -Force

Write-Host "==> Done — sidecar ready at: $DstBin"
