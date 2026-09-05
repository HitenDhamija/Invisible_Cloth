# Blue Invisibility Cloak

![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red)
![Tests](https://img.shields.io/badge/Tests-386-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time invisibility effect built with Python and OpenCV. Hold a blue cloth in front of your webcam and watch yourself disappear — the camera shows the background where the cloth was, creating a Harry Potter-style invisibility cloak.

---

## How It Works

1. **Capture** — The app records your empty background (3 seconds)
2. **Detect** — Every frame, it finds all blue pixels using HSV color segmentation
3. **Replace** — Blue areas are replaced with the saved background
4. **Result** — The cloth becomes invisible on screen

The system uses a **7-stage mask refinement pipeline** (noise removal, hole filling, edge smoothing) and **temporal smoothing** to eliminate flicker — producing a clean, stable effect in real time.

---

## Demo

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  You + Blue  │ ───▶ │  Detect Blue │ ───▶ │  Show Result │
│    Cloth     │      │    Pixels    │      │   Invisible  │
└──────────────┘      └──────────────┘      └──────────────┘
```

---

## Key Features

| Feature | What It Does |
|---------|-------------|
| Real-time processing | 30+ FPS webcam input, zero lag |
| 3 detection modes | HSV (fast), Person-Aware, AI Hybrid (accurate) |
| Auto-calibration | Automatically tunes HSV thresholds to your cloth |
| Adaptive preprocessing | Handles varying lighting with CLAHE |
| Temporal smoothing | Eliminates flicker across frames |
| Soft & hard blending | Two compositing modes for natural output |
| Video recording | Save your sessions as MP4 |
| Screenshots | Capture any frame instantly |
| Fullscreen mode | Immersive display |
| 50+ config options | Fine-tune everything via YAML or CLI |

---

## Technologies

| Technology | Role |
|------------|------|
| **Python 3.11+** | Core language |
| **OpenCV** | Image processing, camera capture, display |
| **NumPy** | Fast pixel-level array operations |
| **Pydantic** | Type-safe configuration validation |
| **PyYAML** | Config file loading |
| **MediaPipe** | Person pose detection (optional) |
| **ONNX Runtime** | AI model inference (optional) |

### Why HSV Instead of RGB?

RGB mixes color and brightness together. HSV separates **color (Hue)** from **brightness (Value)**, so a blue pixel stays "blue" whether it's in sunlight or shadow. This makes detection far more reliable.

---

## Architecture

```
Camera Frame
     │
     ▼
┌─────────────────────┐
│  Background Capture  │  Median of 30 frames → clean background
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Adaptive Preprocessor│  CLAHE + brightness normalization
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Blue Detection      │  HSV segmentation → raw mask
│  (3 modes available) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  7-Stage Refinement  │  Blur → Morph Open → Close → Dilate
│                      │  → Erode → Contour Filter → Soft Mask
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Temporal Smoothing  │  EMA + persistence → flicker-free mask
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Rendering           │  Replace blue regions with background
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Display + Record    │  Fullscreen window, optional MP4 save
└─────────────────────┘
```

### Detection Modes

| Mode | Speed | AI Required | Best For |
|------|-------|-------------|----------|
| **HSV** | 2600+ FPS | No | Fast, single-color detection |
| **Person-Aware** | 660+ FPS | MediaPipe | Multiple people, background rejection |
| **AI Hybrid** | 100+ FPS | YOLOv8 | Maximum accuracy, zero false positives |

The **AI Hybrid** mode combines HSV detection with YOLOv8 person segmentation — it only detects blue pixels that are on a person, ignoring blue backgrounds entirely.

---

## Installation

```bash
# Clone
git clone https://github.com/your-username/blue-invisibility-cloak.git
cd blue-invisibility-cloak

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install core dependencies
pip install -e ".[dev]"

# Optional: AI features (MediaPipe + YOLOv8)
pip install -e ".[ai]"
```

### Requirements

- Python >= 3.11
- OpenCV >= 4.8.0
- A working webcam

---

## Usage

### Quick Start

```bash
# Run with defaults (fullscreen welcome screen, then camera)
python -m cloak.main

# Debug mode — shows mask visualization
python -m cloak.main --debug

# Use a specific camera
python -m cloak.main --camera 1

# Run on a video file
python -m cloak.main --video path/to/video.mp4

# AI Hybrid mode (requires onnxruntime + model)
python -m cloak.main --mode ai_hybrid
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **ESC** | Quit |
| **H** | Toggle help panel |
| **B** | Recapture background |
| **P** | Pause / resume |
| **M** | Switch detection mode |
| **C** | Start auto-calibration |
| **A** | Accept calibration |
| **X** | Cancel calibration |
| **V** | Toggle cloak / debug view |
| **F** | Toggle fullscreen |
| **R** | Start/stop recording |
| **S** | Take screenshot |
| **G** | Toggle soft/hard blend |
| **O** | Toggle performance overlay |
| **D** | Toggle background debug |
| **+/-** | Adjust blend alpha |
| **1-8** | Select debug view |

### Auto-Calibration

Perfect for tuning to your specific shade of blue:

1. Press **C** to enter calibration mode
2. Hold your blue cloth in the center ROI box
3. Press **C** again to compute optimal HSV thresholds
4. Press **A** to apply, **X** to cancel

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `--config PATH` | Custom YAML configuration file |
| `--camera INDEX` | Camera device index (default: 0) |
| `--video PATH` | Use a video file instead of webcam |
| `--mode MODE` | `hsv`, `person_aware_hsv`, or `ai_hybrid` |
| `--debug` | Enable debug view and verbose logging |
| `--profile NAME` | Load a saved calibration profile |
| `--no-ai` | Disable AI detection |
| `--resolution WxH` | Camera resolution (e.g. `1280x720`) |
| `--fps N` | Camera FPS |

---

## Project Structure

```
blue-invisibility-cloak/
├── src/cloak/
│   ├── main.py                    # Entry point + main loop
│   ├── app_state.py               # Finite state machine (6 states)
│   ├── camera/webcam.py           # Camera capture
│   ├── capture/model.py           # Background model
│   ├── config/
│   │   ├── schemas.py             # Pydantic config schemas
│   │   └── loader.py              # YAML config loader
│   ├── detection/
│   │   ├── detector.py            # HSV color detection
│   │   ├── segmenter.py           # AI hybrid detection
│   │   ├── person.py              # MediaPipe wrapper
│   │   ├── person_aware.py        # Person-aware detection
│   │   ├── auto_calibrator.py     # Auto HSV calibration
│   │   ├── adaptive.py            # Adaptive preprocessing
│   │   └── profile_manager.py     # Calibration profiles
│   ├── processing/
│   │   ├── refiner.py             # 7-stage mask refinement
│   │   └── temporal.py            # Temporal smoothing
│   ├── rendering/renderer.py      # Invisibility compositing
│   ├── recording/
│   │   ├── recorder.py            # Video recording
│   │   └── screenshot.py          # Screenshot capture
│   ├── monitoring/performance.py  # FPS / timing tracker
│   ├── ui/
│   │   ├── help_overlay.py        # Keyboard shortcuts panel
│   │   └── error_display.py       # On-screen error messages
│   └── utils/logging.py           # Logging setup
├── configs/
│   └── default.yaml               # Full configuration (50+ options)
├── tests/                         # 386 unit + integration tests
├── benchmarks/                    # Performance benchmarks
├── docs/                          # Algorithms, AI pipeline, evaluation
└── pyproject.toml                 # Build config + dependencies
```

---

## Configuration

All settings live in `configs/default.yaml`. Key options:

```yaml
camera:
  device_id: 0
  width: 640
  height: 480
  fps: 30

detection:
  mode: hsv                    # hsv | person_aware_hsv | ai_hybrid
  hsv_lower: [85, 50, 50]     # Blue lower bound (H, S, V)
  hsv_upper: [135, 255, 255]  # Blue upper bound (H, S, V)

rendering:
  use_soft_blend: true         # Soft feathered edges
  blend_alpha: 0.7             # Blend strength

temporal:
  enabled: true
  ema_alpha: 0.7               # Smoothing strength
  persistence_frames: 2        # Min frames before pixel appears
```

---

## Testing

```bash
# Run all 386 tests
pytest

# Run with coverage
pytest --cov=cloak --cov-report=term-missing

# Run specific test file
pytest tests/test_detection.py
```

Tests use mocked camera input — no webcam required.

---

## Windows Camera Troubleshooting

If the camera doesn't open:

1. **Settings** > **Privacy & security** > **Camera** — ensure camera access is ON
2. Close other apps using the camera (Teams, Zoom, Discord, browser tabs)
3. Try a different camera index: `python -m cloak.main --camera 1`
4. Use DirectShow backend: set `OPENCV_VIDEOIO_PRIORITY_DSHOW=1` before running

---

## License

MIT License — see [LICENSE](LICENSE) for details.
