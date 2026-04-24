#!/usr/bin/env bash
# build_and_render.sh - End-to-end pipeline:
#   1. compile the C++ renderer
#   2. render the PPM frame sequence
#   3. stitch the frames into an MP4 with ffmpeg
#
# Usage:  ./build_and_render.sh
# Output: build/penguin_walk.mp4

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BUILD_DIR="build"
FRAMES_DIR="$BUILD_DIR/frames"
BIN="$BUILD_DIR/penguin_walk"
OBJ="assets/penguin.obj"
MP4="$BUILD_DIR/penguin_walk.mp4"

mkdir -p "$BUILD_DIR"

echo "[1/3] Compiling..."
# Header-only dependencies => one command is enough; no CMake needed.
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wno-unused-parameter \
    src/main.cpp -o "$BIN"

echo "[2/3] Rendering PPM frames..."
rm -rf "$FRAMES_DIR"
mkdir -p "$FRAMES_DIR"
"$BIN" "$OBJ" "$FRAMES_DIR"

echo "[3/3] Encoding MP4 with ffmpeg..."
# -framerate applies to input; -r applies to output.  yuv420p is needed for
# broad player compatibility (QuickTime, Windows Media, web browsers).
ffmpeg -y -loglevel error \
    -framerate 30 -i "$FRAMES_DIR/frame_%04d.ppm" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    -movflags +faststart \
    "$MP4"

echo "Done -> $MP4"
