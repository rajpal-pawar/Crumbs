#!/bin/bash
# build-production.sh
# Automates the v1.0.0 production build pipeline for Linux (DEB/AppImage)

set -e

echo -e "\e[1;36m========================================\e[0m"
echo -e "\e[1;36m Crumbs v1.0.0 - Linux Build Pipeline\e[0m"
echo -e "\e[1;36m========================================\e[0m"
echo ""

DAEMON_PATH="../src-tauri/binaries/crumbs-daemon-x86_64-unknown-linux-gnu"
if [ ! -f "$DAEMON_PATH" ]; then
    echo -e "\e[1;31m❌ Error: Backend daemon binary not found at:\e[0m"
    echo -e "\e[1;31m   $DAEMON_PATH\e[0m"
    echo -e "\e[1;33mPlease compile the backend daemon first before bundling.\e[0m"
    exit 1
fi
echo -e "\e[1;32m✅ Backend daemon binary validated.\e[0m"

echo -e "\e[1;36m📦 Verifying and installing frontend dependency layers...\e[0m"
npm install

echo -e "\e[1;36m🚀 Booting native Tauri build sequence (DEB/AppImage)...\e[0m"
cd ..
./ui/node_modules/.bin/tauri build

echo ""
echo -e "\e[1;32m🎉 Build Complete!\e[0m"
echo -e "\e[1;36mYour compiled .deb and .AppImage installers are successfully bundled and saved in:\e[0m"
echo -e "\e[1;37m-> src-tauri/target/release/bundle\e[0m"
