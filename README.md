<div align="center">
  <img src="ui/public/logo-transparent.png" alt="Crumbs Logo" width="220" />
 
  <p><strong>Your intelligent, on-device file search engine.</strong></p>
  
  <p>
    <a href="https://github.com/rajpal-pawar/Crumbs/releases/latest"><strong>Download for Windows</strong></a> · 
    <a href="https://github.com/rajpal-pawar/Crumbs/releases/latest"><strong>Download for Linux</strong></a>
  </p>
</div>

<br/>

## 🔍 About Crumbs

**Crumbs** is a lightning-fast, privacy-first local desktop search application built for developers, researchers, and power users. Instead of relying on rigid exact-match keywords or uploading your private files to the cloud, Crumbs runs a localized **Artificial Intelligence pipeline directly on your machine**. 

It intelligently reads your documents, code, PDFs, and images, indexing them using advanced semantic embeddings (MiniLM & CLIP). This allows you to find exactly what you're looking for by searching with natural language concepts rather than specific filenames.

### 🌟 Key Features
- **🧠 Semantic & Hybrid Search**: Search by meaning and concepts. Can't remember a file name? Just describe what's inside it!
- **⚡ Offline & Privacy First**: Your files never leave your computer. 100% of the indexing and neural network inference runs securely on-device.
- **🖼️ Multi-modal Support**: Understands code (`.rs`, `.ts`, `.py`), documents (`.pdf`, `.md`, `.txt`), and even images (`.png`, `.jpg`).
- **🚀 Ultra-Low Overhead**: Built natively in Rust via Tauri for minimal RAM and CPU footprint.
- **🎯 Lightning Quick Access**: Instantly summon the search bar with `Ctrl + Shift + Space` from anywhere on your OS.
- **⚙️ OS Integration**: Launches silently on boot so your files are always indexed and ready.

---

## 📥 Installation

You can download the compiled native installers for Windows and Linux directly from the [Releases Page](https://github.com/rajpal-pawar/Crumbs/releases).

| Operating System | Installer Download |
| :--- | :--- |
| **Windows (x64)** | [🔽 Download `.exe` Installer](https://github.com/rajpal-pawar/Crumbs/releases/latest) |
| **Linux (Debian/Ubuntu)** | [🔽 Download `.deb` Package](https://github.com/rajpal-pawar/Crumbs/releases/latest) |
| **Linux (AppImage)** | [🔽 Download `.AppImage`](https://github.com/rajpal-pawar/Crumbs/releases/latest) |

*Note: MacOS support is currently in development and will be available in future releases.*

---

## 🛠️ Tech Stack

Crumbs is engineered for maximum performance and a beautiful user experience:
* **Frontend**: React 19, TypeScript, Vite, Vanilla CSS
* **Backend**: Rust 🦀
* **Framework**: Tauri v2
* **Database**: SQLite (with FTS5 text search integration)
* **Machine Learning**: 
  * `ort` (ONNX Runtime) for running quantized models
  * `MiniLM-L6-v2` for dense text embeddings
  * `CLIP-ViT-B-32` for image vectorization
  * `Pdfium` & `ocrs` for PDF text extraction and OCR

---

## 🏗️ Development Setup

If you want to build Crumbs from source, you'll need [Rust](https://rustup.rs/) and [Node.js](https://nodejs.org/) installed on your machine.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rajpal-pawar/Crumbs.git
   cd Crumbs
   ```

2. **Install UI dependencies**:
   ```bash
   cd ui
   npm install
   ```

3. **Run in development mode**:
   ```bash
   npm run tauri dev
   ```

4. **Build for production**:
   ```bash
   npm run tauri build
   ```

---

<div align="center">
  <i>Built with ❤️ using Rust and Tauri.</i>
</div>
