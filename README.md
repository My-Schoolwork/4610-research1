# Cubism Penguin Walk Cycle

A real-time 3D penguin walk cycle renderer built with a software rasterizer and SFML for display. The penguin is constructed from transformed unit cubes arranged in a hierarchical skeleton.

## Prerequisites

- C++17 compiler (g++ or Clang)
- CMake 3.14+
- SFML 2.5+ (`apt install libsfml-dev` / `brew install sfml` / `vcpkg install sfml`)
- ffmpeg (only needed for MP4 export)

## Building

```bash
mkdir build && cd build
cmake ..
make
```

This produces two executables in `build/`:

- `penguin_walk` — Walk cycle animation viewer with joint ROM overlay
- `penguin_walk_interactive` — WASD-controlled interactive viewer with multiple camera angles

## Running

### Walk Cycle Viewer (`penguin_walk`)

```bash
./penguin_walk ../assets/penguin.obj
```
    
Controls:
- Space — pause/resume
- Left/Right — slow down/speed up animation
- R — reset to start
- J — toggle joint ROM arc overlay
- Esc/Q — quit

### Interactive Viewer (`penguin_walk_interactive`)

```bash
./penguin_walk_interactive ../assets/penguin.obj
```

Controls:
- W/S — walk forward/backward
- A/D — turn left/right
- F — toggle walk/run mode
- 0 — third-person follow camera (default)
- 1–5 — fixed cameras (Resident Evil style)
- Space — pause/resume
- R — reset position and orientation
- Esc/Q — quit