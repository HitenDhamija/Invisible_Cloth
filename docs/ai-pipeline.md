# AI Pipeline Documentation

Technical overview of the AI-augmented detection modes in the Blue Invisibility
Cloak project.

---

## Table of Contents

1. [Detection Mode Architecture](#1-detection-mode-architecture)
2. [AI Hybrid Detection (YOLOv8n-seg ONNX)](#2-ai-hybrid-detection-yolov8n-seg-onnx)
3. [Person-Aware Detection (MediaPipe)](#3-person-aware-detection-mediapipe)
4. [Model Management and Auto-Download](#4-model-management-and-auto-download)
5. [Inference Optimization](#5-inference-optimization)

---

## 1. Detection Mode Architecture

The system supports three detection modes, selectable at runtime:

| Mode | Class | AI Dependency | Description |
|------|-------|---------------|-------------|
| `hsv` | `BlueColorDetector` | None | HSV-only color segmentation |
| `person_aware_hsv` | `PersonAwareDetector` | mediapipe | HSV + MediaPipe pose landmarks |
| `ai_hybrid` | `AIHybridDetector` | onnxruntime | HSV + ONNX person segmentation |

All modes share a common `detect(frame) -> (mask, stats)` interface defined by
the `SegmenterProtocol`.

### Why Not Just Use AI?

Neither method alone is sufficient:

- **HSV alone** false-positives on blue backgrounds (posters, sky, furniture,
  blue shirts on other people).
- **AI alone** cannot distinguish blue cloth from other clothing -- it only
  knows "person."

The **hybrid approach** combines both:

```
frame
    |-- AI model  ->  person_mask (binary, full frame)
    |-- HSV det   ->  blue_mask   (binary, full frame)
    \-- AND       ->  final_mask  (blue only on person)
```

**Spatial prior (person) + color prior (blue) = robust detection.**

---

## 2. AI Hybrid Detection (YOLOv8n-seg ONNX)

### Model

The hybrid detector uses **YOLOv8n-seg** (nano segmentation variant) exported
to ONNX format. This model performs instance segmentation and can produce
per-pixel person masks.

- **Architecture:** YOLOv8-nano with segmentation head
- **Input:** 320x240 BGR frame (configurable via `inference_width`/`inference_height`)
- **Output:** Person class (class 0 in COCO) segmentation mask
- **Format:** ONNX (cross-platform, no PyTorch dependency at inference)

### Per-Frame Pipeline

```
Frame (BGR, H x W x 3)
    |
    v
[1] HSV Blue Detection (always runs)
    |-> blue_mask (uint8, 0 or 255)
    |
    v
[2] AI Person Segmentation (with frame skipping)
    |-> person_mask (uint8, 0 or 255)
    |
    v
[3] Intersection: blue_mask AND person_mask
    |-> final_mask
    |
    v
[4] Fallback: if no person detected, return pure HSV mask
```

### Fallback Behavior

When `fallback_to_hsv` is `true` (default), the system gracefully degrades:

- If the AI model fails to load (missing onnxruntime, model file not found)
- If no person is detected in the frame
- If inference throws an exception

In all cases, the pure HSV mask is returned with a logged warning.

### Output Postprocessing

The ONNX model output is handled generically. Two common formats are supported:

1. **Semantic segmentation output** (shape `(1, C, H, W)` or `(1, 1, H, W)`):
   - Extract the person class channel (index 0)
   - Threshold at `confidence_threshold` (default 0.5)

2. **YOLO detection output** (shape `(1, 4+nc, num_anchors)`):
   - First 4 channels: box coordinates (cx, cy, w, h) normalized to [0, 1]
   - Remaining channels: class confidence scores
   - Filter anchors where class 0 (person) confidence exceeds threshold
   - Generate spatial activation map from bounding boxes

The resulting probability map is resized to the original frame dimensions and
thresholded to produce a binary mask.

---

## 3. Person-Aware Detection (MediaPipe)

### Model

Uses **MediaPipe Pose Landmarker** for lightweight person detection without a
full segmentation model.

- **Models available:**
  - `pose_landmarker_lite.task` (complexity 0, fastest)
  - `pose_landmarker_full.task` (complexity 1, balanced)
  - `pose_landmarker_heavy.task` (complexity 2, most accurate)
- **Output:** Person segmentation mask (float32, values 0.0-1.0)
- **Tracking:** Supports temporal tracking across frames via timestamps

### Per-Frame Pipeline

```
Frame (BGR, H x W x 3)
    |
    v
[1] HSV Blue Detection (always runs)
    |-> blue_mask
    |
    v
[2] MediaPipe Pose Detection
    |-> person_mask (float32, 0.0-1.0)
    |
    v
[3] Threshold person_mask at person_threshold (default 0.5)
    |-> person_binary (uint8, 0 or 255)
    |
    v
[4] Intersection: blue_mask AND person_binary
    |-> final_mask
    |
    v
[5] Fallback: if no person detected, return pure HSV mask
```

### Lazy Initialization

MediaPipe is imported lazily to avoid loading the AI dependency when
person-aware mode is disabled. The model is downloaded automatically on first
use if not present locally.

---

## 4. Model Management and Auto-Download

### ModelManager

The `ModelManager` class handles all AI model lifecycle operations:

```
ModelManager
    |-- ensure_loaded()    -- lazy-load on first inference
    |-- predict()          -- run inference with preprocessing
    |-- device_info        -- reports CPU/GPU availability
    \-- close()            -- release session resources
```

### Model Resolution

Models are resolved in this order:

1. **Explicit path** (`hybrid_model_path` in config)
2. **Default locations:**
   - `<project_root>/models/yolov8n-seg.onnx`
   - `<cwd>/models/yolov8n-seg.onnx`
   - `<cwd>/yolov8n-seg.onnx`
3. **Auto-download** from Ultralytics GitHub releases

### Auto-Download

If the model is not found locally, the system attempts to download it automatically:

```
Source: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.onnx
Dest:   <project_root>/models/yolov8n-seg.onnx
```

**Important Note:** The ONNX model may not be available for direct download from
the Ultralytics assets repository. In this case, you must export the model manually:

```bash
pip install ultralytics
yolo export model=yolov8n-seg.pt imgsz=640 format=onnx opset=12 simplify
```

Then place the exported `yolov8n-seg.onnx` in the `models/` directory or set
`hybrid_model_path` in your config to point to the exported model.

### Device Selection

ONNX Runtime providers are queried and selected automatically:

```
Available providers: ['CPUExecutionProvider', 'CUDAExecutionProvider']
-> Selected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

- **GPU preferred:** CUDA > TensorRT > CPU
- **CPU fallback:** Always available
- **Session options:** Graph optimization enabled, 1 inter-op thread, 2 intra-op
  threads

### MediaPipe Model Download

MediaPipe pose models are downloaded from Google Cloud Storage:

```
Source: https://storage.googleapis.com/mediapipe-models/pose_landmarker/{model_name}/float16/latest/{model_name}
Dest:   <project_root>/models/{model_name}
```

---

## 5. Inference Optimization

Running AI inference every frame is expensive. Three optimization strategies are
implemented:

### 5.1 Frame Skipping

Run AI inference only every $N$ frames and cache the result:

```python
self._frame_counter += 1
skip = self._ai_cfg.inference_frame_skip

if self._frame_counter % skip == 0 or self._cached_person_mask is None:
    self._cached_person_mask = self._model_manager.predict(frame)
```

**Configuration:**
- `inference_frame_skip`: Run AI every N frames (default 1, range 1-10)
- Setting to 3 means AI runs at ~10 FPS on a 30 FPS stream

### 5.2 Resolution Reduction

AI inference runs at a lower resolution than the display frame:

```
Display frame:  640 x 480 (full resolution)
AI inference:   320 x 240 (configurable)
```

**Configuration:**
- `inference_width`: Inference width (default 320, range 160-1920)
- `inference_height`: Inference height (default 240, range 120-1080)

Lower resolution dramatically reduces inference time with minimal impact on
person segmentation quality, since the mask is resized back to full resolution
using nearest-neighbor interpolation.

### 5.3 Half Precision (FP16)

On supported GPUs, FP16 inference can approximately double throughput:

```python
if config.use_half_precision and "CUDA" in device_info:
    sess_opts.enable_cpu_mem_arena = False  # reduces memory
```

**Configuration:**
- `use_half_precision`: Enable FP16 inference (default false, GPU only)

### Combined Strategy

The default configuration balances quality and performance:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `inference_frame_skip` | 1 | Run every frame |
| `inference_width` | 320 | Quarter resolution |
| `inference_height` | 240 | Quarter resolution |
| `use_half_precision` | false | FP32 inference |

For lower-end hardware, increase `inference_frame_skip` to 2-3 and reduce
resolution further (e.g., 160x120).

### Latency Tracking

Each inference call records its latency:

```python
t0 = time.perf_counter()
outputs = self._session.run(None, {self._input_name: blob})
self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
```

This is exposed via `model_manager.last_latency_ms` for benchmarking and
debugging.

---

## Architecture Diagram

```
+------------------+     +-------------------+     +------------------+
|  BlueColorDetector|     | PersonAwareDetector|     | AIHybridDetector |
|  (HSV only)      |     | (MediaPipe + HSV)  |     | (ONNX + HSV)     |
+--------+---------+     +---------+---------+     +--------+---------+
         |                         |                         |
         v                         v                         v
  detect(frame)            detect(frame)              detect(frame)
         |                         |                         |
         v                         v                         v
   HSV threshold          HSV + pose mask           HSV + person mask
         |                         |                         |
         v                         v                         v
   binary mask            bitwise AND mask          bitwise AND mask
         |                         |                         |
         +--------+--------+-------+--------+--------+-------+
                  |        |                |        |
                  v        v                v        v
            +------------------------------------------+
            |         SegmenterProtocol                |
            |   detect(frame) -> (mask, stats)         |
            +------------------------------------------+
```
