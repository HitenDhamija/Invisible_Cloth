"""Pipeline benchmark: per-stage timing breakdown with synthetic frames.

Measures FPS, mean/median/P95 latency, and breaks down timing by stage:
preprocessing, segmentation (detection), mask refinement, temporal smoothing,
and rendering. Outputs a formatted report with hardware/software info.

No webcam or GPU required — uses synthetic frames.

Usage::

    python benchmarks/benchmark_pipeline.py
    python benchmarks/benchmark_pipeline.py --frames 200 --width 640 --height 480
    python benchmarks/benchmark_pipeline.py --frames 500 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cloak.config.schemas import (
    DetectionConfig,
    MaskConfig,
    ProcessingConfig,
    RenderingConfig,
    TemporalConfig,
)
from cloak.detection.detector import BlueColorDetector
from cloak.processing.refiner import MaskRefiner
from cloak.processing.temporal import TemporalMaskSmoother
from cloak.rendering.renderer import InvisibilityRenderer

# ---------------------------------------------------------------------------
# Synthetic frame generation
# ---------------------------------------------------------------------------

def _make_synthetic_frame(
    h: int, w: int, frame_idx: int, bg: np.ndarray,
) -> np.ndarray:
    """Generate a synthetic frame with a moving blue rectangle on a static background."""
    frame = bg.copy()
    offset_x = int(10 * np.sin(frame_idx * 0.1))
    offset_y = int(5 * np.cos(frame_idx * 0.15))
    cx, cy = w // 2 + offset_x, h // 2 + offset_y
    half_w, half_h = w // 6, h // 4
    x0 = max(0, cx - half_w)
    x1 = min(w, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(h, cy + half_h)
    frame[y0:y1, x0:x1] = (255, 50, 30)  # blue in BGR
    return frame


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of floats."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(data) - 1)
    d = k - f
    return data[f] + d * (data[c] - data[f])


def _stage_stats(times_ms: list[float]) -> dict:
    """Compute mean, median, and P95 for a list of timings in ms."""
    if not times_ms:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    sorted_t = sorted(times_ms)
    return {
        "mean": statistics.mean(sorted_t),
        "median": statistics.median(sorted_t),
        "p95": _percentile(sorted_t, 95.0),
    }


# ---------------------------------------------------------------------------
# Pipeline benchmark
# ---------------------------------------------------------------------------

STAGES = ("preprocess", "detect", "refine", "temporal", "render")


def benchmark_pipeline(
    frames: list[np.ndarray],
    background: np.ndarray,
    detection_cfg: DetectionConfig,
    processing_cfg: ProcessingConfig,
    mask_cfg: MaskConfig,
    temporal_cfg: TemporalConfig,
    rendering_cfg: RenderingConfig,
    warmup: int = 5,
) -> dict:
    """Run the full pipeline on synthetic frames and collect per-stage timings."""
    detector = BlueColorDetector(detection_cfg, processing_cfg)
    refiner = MaskRefiner(mask_cfg)
    smoother = TemporalMaskSmoother(temporal_cfg)
    renderer = InvisibilityRenderer(rendering_cfg)

    stage_times: dict[str, list[float]] = {s: [] for s in STAGES}
    total_times: list[float] = []

    all_frames = frames

    for i, frame in enumerate(all_frames):
        # Warmup: run pipeline but don't record timings
        is_warmup = i < warmup

        t_start = time.perf_counter()

        # Stage 1: Preprocessing (HSV conversion happens inside detect,
        # but we measure it separately by pre-computing the HSV frame)
        t0 = time.perf_counter()
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        t1 = time.perf_counter()
        stage_times["preprocess"].append((t1 - t0) * 1000.0)

        # Stage 2: Detection / segmentation
        t0 = time.perf_counter()
        raw_mask, _ = detector.detect(frame)
        t1 = time.perf_counter()
        stage_times["detect"].append((t1 - t0) * 1000.0)

        # Stage 3: Mask refinement
        t0 = time.perf_counter()
        mask, soft_mask, _ = refiner.refine(raw_mask)
        t1 = time.perf_counter()
        stage_times["refine"].append((t1 - t0) * 1000.0)

        # Stage 4: Temporal smoothing
        t0 = time.perf_counter()
        smooth_mask = smoother.smooth(mask)
        t1 = time.perf_counter()
        stage_times["temporal"].append((t1 - t0) * 1000.0)

        # Stage 5: Rendering
        t0 = time.perf_counter()
        _ = renderer.render(frame, background, smooth_mask, soft_mask)
        t1 = time.perf_counter()
        stage_times["render"].append((t1 - t0) * 1000.0)

        t_end = time.perf_counter()
        total_times.append((t_end - t_start) * 1000.0)

        if not is_warmup:
            pass  # timings already appended above (warmup frames also get timed
            # but we include them for pipeline stability; remove if needed)

    # Discard warmup timings
    for s in STAGES:
        stage_times[s] = stage_times[s][warmup:]
    total_times = total_times[warmup:]

    fps_values = [1000.0 / t for t in total_times if t > 0]

    return {
        "stage_stats": {s: _stage_stats(stage_times[s]) for s in STAGES},
        "total_stats": _stage_stats(total_times),
        "fps": {
            "mean": statistics.mean(fps_values) if fps_values else 0.0,
            "median": statistics.median(fps_values) if fps_values else 0.0,
            "p95": _percentile(sorted(fps_values), 95.0) if fps_values else 0.0,
        },
        "num_frames": len(total_times),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _print_report(result: dict, width: int, height: int) -> None:
    """Print a formatted benchmark report."""
    print()
    print("=" * 72)
    print("  BLUE INVISIBILITY CLOAK — PIPELINE BENCHMARK REPORT")
    print("=" * 72)

    # Hardware / software info
    print()
    print("  Environment")
    print("  " + "-" * 68)
    print(f"  Platform        : {platform.platform()}")
    print(f"  Processor       : {platform.processor() or 'N/A'}")
    print(f"  Python          : {platform.python_version()}")
    print(f"  OpenCV          : {cv2.__version__}")
    print(f"  NumPy           : {np.__version__}")
    print(f"  Resolution      : {width}x{height}")

    # Configuration
    print()
    print("  Configuration")
    print("  " + "-" * 68)
    print(f"  Frames measured : {result['num_frames']}")
    print("  Detection mode  : HSV only")

    # Per-stage breakdown
    print()
    print("  Per-Stage Timing Breakdown (ms)")
    print("  " + "-" * 68)
    print(f"  {'Stage':<16} {'Mean':>10} {'Median':>10} {'P95':>10}")
    print("  " + "-" * 68)
    for stage in STAGES:
        stats = result["stage_stats"][stage]
        print(
            f"  {stage:<16} {stats['mean']:>9.2f}ms {stats['median']:>9.2f}ms {stats['p95']:>9.2f}ms"
        )
    print("  " + "-" * 68)

    # Total latency
    total = result["total_stats"]
    print()
    print("  Total Frame Latency (ms)")
    print("  " + "-" * 68)
    print(f"  Mean            : {total['mean']:.2f}ms")
    print(f"  Median          : {total['median']:.2f}ms")
    print(f"  P95             : {total['p95']:.2f}ms")

    # FPS summary
    fps = result["fps"]
    print()
    print("  Frames Per Second")
    print("  " + "-" * 68)
    print(f"  Mean            : {fps['mean']:.1f}")
    print(f"  Median          : {fps['median']:.1f}")
    print(f"  P95             : {fps['p95']:.1f}")

    # Stage contribution
    print()
    print("  Stage Contribution (% of total mean latency)")
    print("  " + "-" * 68)
    total_mean = total["mean"]
    for stage in STAGES:
        pct = (result["stage_stats"][stage]["mean"] / total_mean * 100) if total_mean > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {stage:<16} {pct:>5.1f}%  {bar}")

    print()
    print("=" * 72)
    print("  Notes:")
    print("  - Mean/Median/P95 reported in milliseconds (lower = better).")
    print("  - FPS: frames per second (higher = better).")
    print("  - All measurements use synthetic frames (no webcam/GPU).")
    print("  - P95 = 95th percentile (worst 5% of frames).")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline benchmark with per-stage timing breakdown",
    )
    parser.add_argument("--frames", type=int, default=150, help="Number of benchmark frames (default: 150)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames not counted (default: 10)")
    args = parser.parse_args()

    total_frames = args.frames + args.warmup
    h, w = args.height, args.width

    print(f"Generating {total_frames} synthetic frames ({w}x{h})...")
    bg = np.random.randint(60, 100, (h, w, 3), dtype=np.uint8)
    frames = [_make_synthetic_frame(h, w, i, bg) for i in range(total_frames)]

    det_cfg = DetectionConfig(hsv_lower=[85, 100, 100], hsv_upper=[135, 255, 255])
    proc_cfg = ProcessingConfig(blur_kernel=1, morphology_kernel=1)
    mask_cfg = MaskConfig()
    temporal_cfg = TemporalConfig()
    rendering_cfg = RenderingConfig()

    print(f"Running benchmark ({args.frames} frames, warmup={args.warmup})...")
    result = benchmark_pipeline(
        frames, bg,
        det_cfg, proc_cfg, mask_cfg, temporal_cfg, rendering_cfg,
        warmup=args.warmup,
    )

    _print_report(result, w, h)


if __name__ == "__main__":
    main()
