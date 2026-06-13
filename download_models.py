#!/usr/bin/env python3
"""
download_models.py — Download INT8-quantized ONNX models for Crumbs.

Downloads three files from the Xenova namespace on HuggingFace into:
  - ./models/  (project root, for development convenience)
  - <platform data dir>/crumbs/models/  (runtime location the daemon reads)

Usage:
    python3 download_models.py

No pip dependencies required — uses stdlib urllib only.
"""

import os
import sys
import shutil
import urllib.request

# ---------------------------------------------------------------------------
# Files to download
# ---------------------------------------------------------------------------

DOWNLOADS = [
    {
        "url":      "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
        "filename": "tokenizer.json",
        "desc":     "MiniLM tokenizer",
        "size_hint": "~0.2 MB",
    },
    {
        "url":      "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx",
        "filename": "minilm-l6-int8.onnx",
        "desc":     "MiniLM-L6 INT8 text encoder",
        "size_hint": "~23 MB",
    },
    {
        "url":      "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx",
        "filename": "clip-vision-int8.onnx",
        "desc":     "CLIP ViT-B/32 INT8 visual encoder",
        "size_hint": "~87 MB",
    },
    {
        "url":      "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_quantized.onnx",
        "filename": "clip-text-int8.onnx",
        "desc":     "CLIP ViT-B/32 INT8 text encoder (For searching images)",
        "size_hint": "~60 MB",
    },
    {
        "url":      "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
        "filename": "clip-tokenizer.json",
        "desc":     "CLIP tokenizer",
        "size_hint": "~1 MB",
    },
]

# ---------------------------------------------------------------------------
# Platform data directory helper
# ---------------------------------------------------------------------------

def _get_runtime_dir() -> str:
    """
    Return the runtime model directory that crumbs-daemon reads.
    Mirrors config.rs's model_cache_dir() logic.
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        # Linux: XDG_DATA_HOME or ~/.local/share
        base = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share"))

    return os.path.join(base, "crumbs", "models")

# ---------------------------------------------------------------------------
# Destination directories
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEV_DIR     = os.path.join(SCRIPT_DIR, "models")          # project root/models/
RUNTIME_DIR = _get_runtime_dir()                           # platform data dir


# ---------------------------------------------------------------------------
# Progress hook
# ---------------------------------------------------------------------------

def _make_progress_hook(filename: str, size_hint: str):
    """Returns a urllib reporthook that prints download progress."""
    prev_pct = [-1]

    def hook(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, int(downloaded * 100 / total_size))
            if pct != prev_pct[0]:
                bar_filled = pct // 5
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                mb_done  = downloaded / 1_048_576
                mb_total = total_size / 1_048_576
                print(
                    f"\r  [{bar}] {pct:3d}%  {mb_done:.1f} / {mb_total:.1f} MB",
                    end="", flush=True
                )
                prev_pct[0] = pct
        else:
            # Unknown size — just show bytes downloaded
            mb = downloaded / 1_048_576
            print(f"\r  Downloaded {mb:.1f} MB ({size_hint} expected)…", end="", flush=True)

    return hook


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: str, desc: str, size_hint: str) -> None:
    """
    Download `url` to `dest_path`, printing a progress bar.
    Skips if the file already exists and is non-empty.
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        size_mb = os.path.getsize(dest_path) / 1_048_576
        print(f"  ✓ Already present ({size_mb:.1f} MB) — skipping.")
        return

    tmp_path = dest_path + ".tmp"
    try:
        hook = _make_progress_hook(os.path.basename(dest_path), size_hint)
        urllib.request.urlretrieve(url, tmp_path, reporthook=hook)
        print()  # newline after progress bar
        os.replace(tmp_path, dest_path)
    except Exception as exc:
        # Clean up partial download.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Download failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Crumbs Model Downloader")
    print("=" * 50)
    print(f"Development dir : {DEV_DIR}")
    print(f"Runtime dir     : {RUNTIME_DIR}")
    print()

    # Create both directories.
    os.makedirs(DEV_DIR,     exist_ok=True)
    os.makedirs(RUNTIME_DIR, exist_ok=True)

    failed = []

    for item in DOWNLOADS:
        url       = item["url"]
        filename  = item["filename"]
        desc      = item["desc"]
        size_hint = item["size_hint"]

        dev_dest     = os.path.join(DEV_DIR,     filename)
        runtime_dest = os.path.join(RUNTIME_DIR, filename)

        print(f"→ {desc} ({size_hint})")
        print(f"  URL: {url}")

        try:
            # Download into dev dir first (avoids re-downloading for runtime copy).
            download_file(url, dev_dest, desc, size_hint)

            # Copy to runtime dir (daemon reads from here at startup).
            if not os.path.exists(runtime_dest) or \
               os.path.getsize(runtime_dest) != os.path.getsize(dev_dest):
                print(f"  → Copying to runtime dir…")
                shutil.copy2(dev_dest, runtime_dest)
                print(f"  ✓ Copied.")
            else:
                print(f"  ✓ Runtime copy already up-to-date.")

        except RuntimeError as e:
            print(f"  ✗ ERROR: {e}")
            failed.append(filename)

        print()

    # Summary
    print("=" * 50)
    if not failed:
        print("✓ All models downloaded successfully.")
        print()
        print("Files in models/:")
        for f in sorted(os.listdir(DEV_DIR)):
            path = os.path.join(DEV_DIR, f)
            size_mb = os.path.getsize(path) / 1_048_576
            print(f"  {f:<35}  {size_mb:6.1f} MB")
        print()
        print(f"Files in runtime dir ({RUNTIME_DIR}):")
        for f in sorted(os.listdir(RUNTIME_DIR)):
            path = os.path.join(RUNTIME_DIR, f)
            size_mb = os.path.getsize(path) / 1_048_576
            print(f"  {f:<35}  {size_mb:6.1f} MB")
    else:
        print(f"✗ {len(failed)} file(s) failed to download:")
        for f in failed:
            print(f"  - {f}")
        print("\nRe-run the script to retry. HuggingFace sometimes rate-limits")
        print("large file requests — wait a minute and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
