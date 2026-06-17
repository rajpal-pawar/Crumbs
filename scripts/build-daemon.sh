#!/usr/bin/env bash
# build-daemon.sh — Build the crumbs-daemon sidecar for Linux and copy it to
# the path that Tauri's `externalBin` resolver expects.
#
# Usage:
#   ./scripts/build-daemon.sh              # native release build
#   ./scripts/build-daemon.sh --debug      # debug build (faster, larger binary)
#   TARGET=x86_64-unknown-linux-gnu ./scripts/build-daemon.sh
#
# Tauri's externalBin convention:
#   The binary must live at:
#     src-tauri/bin/crumbs-daemon-<target-triple>
#   e.g. src-tauri/bin/crumbs-daemon-x86_64-unknown-linux-gnu
#
# Exit codes:
#   0  — success
#   1  — cargo build failed
#   2  — binary copy failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
BUILD_MODE="release"
CARGO_FLAGS="--release"
for arg in "$@"; do
    case "$arg" in
        --release)
            BUILD_MODE="release"
            CARGO_FLAGS="--release"
            ;;
        --debug)
            BUILD_MODE="debug"
            CARGO_FLAGS=""
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------- #
# Determine target triple
# --------------------------------------------------------------------------- #
if [[ -z "${TARGET:-}" ]]; then
    TARGET="$(rustc -vV 2>/dev/null | sed -n 's/^host: //p')"
fi

echo "==> Building crumbs-daemon (mode=$BUILD_MODE, target=$TARGET)"

# --------------------------------------------------------------------------- #
# Cargo build
# --------------------------------------------------------------------------- #
cd "$ROOT_DIR"
# shellcheck disable=SC2086
cargo build $CARGO_FLAGS --bin crumbs-daemon 2>&1

SRC_BIN="$ROOT_DIR/target/$BUILD_MODE/crumbs-daemon"
DST_DIR="$ROOT_DIR/src-tauri/binaries"
DST_BIN="$DST_DIR/crumbs-daemon-$TARGET"

if [[ "$TARGET" == *"windows"* ]]; then
    SRC_BIN="${SRC_BIN}.exe"
    DST_BIN="${DST_BIN}.exe"
fi

mkdir -p "$DST_DIR"

echo "==> Copying binary"
echo "    $SRC_BIN"
echo "    → $DST_BIN"
cp -f "$SRC_BIN" "$DST_BIN" || { echo "ERROR: copy failed" >&2; exit 2; }

echo "==> Done — sidecar ready at: $DST_BIN"
