# Crumbs

Crumbs is an intelligent, on-device semantic file search engine. It allows you to search through your local files (documents, images, etc.) using natural language, all while keeping your data completely private and offline.

## Features

- **Semantic Search**: Find files based on meaning, not just exact keyword matches.
- **Multimodal**: Supports text and image search using CLIP and MiniLM models.
- **Privacy First**: Everything runs locally on your machine. No data is sent to the cloud.
- **Lightweight**: Optimized to run quietly in the background with minimal resource usage.
- **High-Performance PDF Parsing**: Ingests massive textbook PDFs with near-zero memory footprint using stream-based checksums and dynamic text-limit short-circuiting.
- **Premium UI/UX**: Features a sleek glassmorphism theme, animated dynamic scrollbars, and an efficient scrollable top-10 search results view to minimize clutter.
- **User-Controlled Indexing**: You choose exactly which folders Crumbs should index and monitor.

## Architecture

Crumbs consists of two main components:
- **Frontend**: A sleek, minimal search interface built with Tauri v2, React, and Vite.
- **Daemon**: A powerful Rust background service that handles file watching, text/image extraction, vector embedding, and SQLite database management.

---

## Installation (Windows)

Follow these step-by-step instructions to get Crumbs running on Windows.

### 1. Prerequisites
You must have the following installed on your system before building:
- **Microsoft Visual Studio C++ Build Tools**: Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/). During installation, you MUST check the **"Desktop development with C++"** workload.
- **WebView2**: (Usually pre-installed on Windows 11). If you are on Windows 10, install the WebView2 Runtime from [Microsoft](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).
- **Rust**: Download and run `rustup-init.exe` from [rustup.rs](https://rustup.rs/). Follow the default installation prompts.
- **Node.js**: Download and install Node.js (v18 or higher) from [nodejs.org](https://nodejs.org/).
- **Python**: Install Python 3 from [python.org](https://www.python.org/downloads/) (make sure to check "Add python.exe to PATH" during installation).

### 2. Clone the Repository
Open PowerShell and run:
```powershell
git clone <repository-url>
cd Crumbs
```

### 3. Download the AI Models
Crumbs requires ONNX models to function. A script is provided to download them automatically:
```powershell
python download_models.py
```

### 4. Install Frontend Dependencies
```powershell
cd ui
npm install
cd ..
```

### 5. Build the Backend Daemon
The Tauri frontend expects the backend daemon to be compiled and placed in a specific `binaries` folder with a platform-specific name. Run these commands in PowerShell:

```powershell
# Build the Rust daemon
cd crumbs-daemon
cargo build --release
cd ..

# Create the binaries folder
New-Item -ItemType Directory -Force -Path src-tauri\binaries

# Copy the compiled executable to the Tauri sidecar directory
Copy-Item crumbs-daemon\target\release\crumbs-daemon.exe src-tauri\binaries\crumbs-daemon-x86_64-pc-windows-msvc.exe
```

### 6. Run or Build the App
To run the app in development mode:
```powershell
npm run tauri dev
```
To compile a standalone `.msi` or `.exe` installer for Windows:
```powershell
npm run tauri build
```

---

## Installation (macOS & Linux)

### 1. Prerequisites
- **Rust**: Install via `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Node.js**: Install via your package manager or from [nodejs.org](https://nodejs.org/) (v18+).
- **Python**: Ensure Python 3 is installed (`python3 --version`).
- **OS-Specific Dependencies**: Follow the [Tauri v2 prerequisites guide](https://v2.tauri.app/start/prerequisites/) for your specific Linux distribution or macOS setup (e.g., `build-essential`, `libwebkit2gtk-4.1-dev` on Ubuntu/Debian).

### 2. Clone & Setup
Open your terminal and run:
```bash
git clone <repository-url>
cd Crumbs

# Download AI Models
python3 download_models.py

# Install Frontend Dependencies
cd ui
npm install
cd ..
```

### 3. Build the Backend Daemon
Build the rust daemon and copy it to the sidecar folder.

```bash
# Build the Rust daemon
cd crumbs-daemon
cargo build --release
cd ..

# Create the binaries folder
mkdir -p src-tauri/binaries
```

**Next, copy the binary based on your OS and Architecture:**

**Linux (x86_64):**
```bash
cp crumbs-daemon/target/release/crumbs-daemon src-tauri/binaries/crumbs-daemon-x86_64-unknown-linux-gnu
```

**macOS (Apple Silicon / M1 / M2 / M3):**
```bash
cp crumbs-daemon/target/release/crumbs-daemon src-tauri/binaries/crumbs-daemon-aarch64-apple-darwin
```

**macOS (Intel):**
```bash
cp crumbs-daemon/target/release/crumbs-daemon src-tauri/binaries/crumbs-daemon-x86_64-apple-darwin
```

### 4. Run or Build the App
To run the app in development mode:
```bash
npm run tauri dev
```
To compile a standalone bundle (e.g., `.app` for macOS, `.deb`/`.AppImage` for Linux):
```bash
npm run tauri build
```

---

## Automated Production Build Pipeline (v1.0.0+)

For a completely automated "Fat Installer" build (bundling the daemon, models, PDFium, and frontend into a single distributable), we provide cross-platform build wrapper scripts. **Make sure you have compiled the daemon and downloaded the models first.**

**Windows:**
```powershell
.\ui\build-production.ps1
```

**Linux:**
```bash
cd ui
./build-production.sh
```
These scripts will automatically verify dependencies, trigger the Tauri compiler from the correct directory context, and output your final bundled installers (`.msi`, `.exe`, `.deb`, `.AppImage`) in the `src-tauri/target/release/bundle/` directory. There is also a fully automated GitHub Actions pipeline configured in `.github/workflows/build.yml` for CI/CD cloud compilation.

---

## Usage

1. **Onboarding**: On first launch, you will be greeted with a welcome screen. Click to select the directories you want Crumbs to index (e.g., Documents, Pictures).
2. **Indexing**: Crumbs will begin scanning and indexing your selected folders in the background. You can monitor the progress in the Settings Dashboard.
3. **Searching**: Use the main search bar to query your files using natural language (e.g., "invoice from last month" or "photos of the beach").
4. **Settings**: Open the settings dashboard to manage your watched folders, adjust engine performance limits, or view the current indexing status.

## Version

Current Version: **v1**
