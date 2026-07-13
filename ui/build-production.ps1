# build-production.ps1
# Automates the v1.0.0 production build pipeline for Windows (MSI/NSIS)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Crumbs v1.0.0 - Windows Build Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$DaemonPath = "..\src-tauri\binaries\crumbs-daemon-x86_64-pc-windows-msvc.exe"
if (-Not (Test-Path $DaemonPath)) {
    Write-Host "❌ Error: Backend daemon binary not found at:" -ForegroundColor Red
    Write-Host "   $DaemonPath" -ForegroundColor Red
    Write-Host "Please compile the backend daemon first before bundling." -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Backend daemon binary validated." -ForegroundColor Green

Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Cyan
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: npm install failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "✅ Dependencies installed." -ForegroundColor Green

Write-Host "🚀 Booting native Tauri build sequence (WiX/NSIS)..." -ForegroundColor Cyan
Set-Location ..
.\ui\node_modules\.bin\tauri build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Tauri build failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "🎉 Build Complete!" -ForegroundColor Green
Write-Host "Your compiled .msi and .exe installers are successfully bundled and saved in:" -ForegroundColor Cyan
Write-Host "-> src-tauri\target\release\bundle" -ForegroundColor White
