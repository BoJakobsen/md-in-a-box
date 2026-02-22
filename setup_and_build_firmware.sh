#!/bin/bash
set -e

# Copyright (C) 2026 Bo Jakobsen
# Dept. of Sciences, Roskilde University, Denmark
# Part of "MD-in-a-Box" - published under the MIT License (see LICENSE)

# ============================================================================
# Build Pimoroni MicroPython firmware with custom MD force-calculation module
# ============================================================================
#
# This script clones all required repos, sets up the build environment,
# links the custom user.c into the ulab build tree, and compiles firmware
# for Raspberry Pi Pico 2.
#
# Prerequisites:
#   sudo apt install cmake build-essential gcc-arm-none-eabi ccache python3-pip git
#
# Usage:
#   ./setup_and_build_firmware.sh [board]
#
# Boards: rpi_pico2, rpi_pico2_w (default), pimoroni_pico_plus2
#
# First run clones ~1 GB of repos and builds everything from scratch.
# Subsequent runs do an incremental build (fast).
# To force a clean rebuild, delete firmware_build/build-<board>/ first.
#
# Output: firmware_build/<board>.uf2
# ============================================================================

BOARD=${1:-rpi_pico2_w}

# --- Resolve project root from script location ---
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/firmware_build"

# --- Check prerequisites ---
MISSING=""
for cmd in git cmake ccache arm-none-eabi-gcc python3; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING="$MISSING $cmd"
    fi
done

if [ -n "$MISSING" ]; then
    echo "ERROR: Missing required tools:$MISSING"
    echo ""
    echo "Install with:"
    echo "  sudo apt install cmake build-essential gcc-arm-none-eabi ccache python3-pip git"
    exit 1
fi

# --- Clone pimoroni-pico-rp2350 (board defs + CI scripts) ---
if [ ! -d "$BUILD_DIR/pimoroni-pico-rp2350" ]; then
    echo "=== Cloning pimoroni-pico-rp2350 ==="
    mkdir -p "$BUILD_DIR"
    git clone https://github.com/pimoroni/pimoroni-pico-rp2350.git "$BUILD_DIR/pimoroni-pico-rp2350"
fi

# --- Set environment and source CI build functions ---
cd "$BUILD_DIR"
export CI_PROJECT_ROOT="$BUILD_DIR/pimoroni-pico-rp2350"
export CI_BUILD_ROOT="$BUILD_DIR"
export CI_USE_ENV=1

# Pimoroni CI script runs "pip install littlefs-python" which fails on
# newer Debian/Ubuntu due to PEP 668 externally-managed-environment.
# This env var is the pip-sanctioned workaround for build scripts.
export PIP_BREAK_SYSTEM_PACKAGES=1

source "$CI_PROJECT_ROOT/ci/micropython.sh"

# --- Clone micropython, pimoroni-pico, tools, build mpy-cross ---
if [ ! -d "$BUILD_DIR/micropython" ] || [ ! -d "$BUILD_DIR/pimoroni-pico" ]; then
    echo "=== Running ci_prepare_all (cloning repos + building mpy-cross) ==="
    ci_prepare_all
fi

# --- Enable user module in ulab (upstream defaults to 0) ---
ULAB_H="$BUILD_DIR/pimoroni-pico/micropython/modules/ulab/code/ulab.h"
if grep -q 'ULAB_HAS_USER_MODULE.*\b0\b' "$ULAB_H" 2>/dev/null; then
    echo "=== Enabling ULAB_HAS_USER_MODULE in ulab.h ==="
    sed -i 's/\(ULAB_HAS_USER_MODULE\s*\)(0)/\1(1)/' "$ULAB_H"
fi

# --- Link custom user.c into the ulab build tree ---
ULAB_USER_DIR="$BUILD_DIR/pimoroni-pico/micropython/modules/ulab/code/user"
if [ ! -L "$ULAB_USER_DIR/user.c" ] || \
   [ "$(readlink -f "$ULAB_USER_DIR/user.c")" != "$(readlink -f "$PROJECT_ROOT/user.c")" ]; then
    echo "=== Linking user.c into ulab build tree ==="
    ln -sf "$PROJECT_ROOT/user.c" "$ULAB_USER_DIR/user.c"
    echo "  $(ls -l "$ULAB_USER_DIR/user.c")"
    # New symlink means we need a clean configure
    rm -rf "$BUILD_DIR/build-$BOARD"
fi

# --- Configure (only if build dir missing) ---
if [ ! -d "$BUILD_DIR/build-$BOARD" ]; then
    echo "=== Configuring $BOARD ==="
    ci_cmake_configure "$BOARD"
fi

# --- Build ---
echo "=== Building $BOARD ==="
ci_cmake_build "$BOARD"

echo ""
echo "=== Build complete ==="
echo ""
echo "To flash: copy $BOARD.uf2 to the Pico 2 USB drive (hold BOOTSEL while plugging in)"
