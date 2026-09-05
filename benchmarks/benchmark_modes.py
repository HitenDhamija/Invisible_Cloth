"""Benchmark comparing detection modes: HSV, person-aware HSV, AI hybrid.

Measures engineering metrics (FPS, latency, mask stability) for each
detection mode. Does NOT measure accuracy (no ground truth available).

Usage::

    python benchmarks/benchmark_modes.py
    python benchmarks/benchmark_modes.py --frames 200 --width 640 --height 480

Output:
    Prints a comparison table with average FPS, average latency per stage,
    and approximate mask stability (fraction of frames where mask changed
    less than 5% from previous frame).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cloak.config.schemas import (
    AIConfig,
    DetectionConfig,
    MaskConfig,
    ProcessingConfig,
)
from cloak.detection.detector import BlueColorDetector
from cloak.detection.person_aware import PersonAwareDetector
from cloak.detection.segmenter import AIHybridDetector


def _make_synthetic_frame(
    h: int,
    w: int,
    frame_idx: int,
) -> np.ndarray:
    """Generate a synthetic frame with a moving blue rectangle.

    Simulates a person-like shape with a blue region that shifts
    slightly each frame to test temporal stability.
    """
    frame = np.random.randint(60, 100, (h, w, 3), dtype=np.uint8)

    # Blue rectangle (simulates cloth) with slight motion
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


def _compute_mask_stability(masks: list[np.ndarray]) -> float:
    """Compute fraction of consecutive frame pairs with <5% mask change."""
    if len(masks) < 2:
        return 1.0

    stable_count = 0
    total = len(masks) - 1
    for i in range(total):
        prev = masks[i].astype(np.float32) / 255.0
        curr = masks[i + 1].astype(np.float32) / 255.0
        diff = np.abs(curr - prev).mean()
        if diff < 0.05:
            stable_count += 1

    return stable_count / total


def benchmark_hsv(
    frames: list[np.ndarray],
    detection_cfg: DetectionConfig,
    processing_cfg: ProcessingConfig,
    mask_cfg: MaskConfig,
) -> dict:
    """Benchmark pure HSV detection."""
    detector = BlueColorDetector(detection_cfg, processing_cfg)
    masks: list[np.ndarray] = []
    latencies: list[float] = []

    for frame in frames:
        t0 = time.perf_counter()
        mask, _ = detector.detect(frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        masks.append(mask)

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    stability = _compute_mask_stability(masks)

    return {
        "mode": "HSV only",
        "avg_fps": fps,
        "avg_latency_ms": avg_latency,
        "stability": stability,
        "mask_count": len(masks),
    }


def benchmark_person_aware(
    frames: list[np.ndarray],
    detection_cfg: DetectionConfig,
    processing_cfg: ProcessingConfig,
    ai_cfg: AIConfig,
) -> dict:
    """Benchmark person-aware HSV detection (MediaPipe)."""
    try:
        detector = PersonAwareDetector(detection_cfg, processing_cfg, ai_cfg)
    except Exception as e:
        return {
            "mode": "Person-aware HSV",
            "avg_fps": 0.0,
            "avg_latency_ms": 0.0,
            "stability": 0.0,
            "mask_count": 0,
            "error": str(e),
        }

    masks: list[np.ndarray] = []
    latencies: list[float] = []

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        mask, _ = detector.detect(frame, timestamp_ms=i * 33)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        masks.append(mask)

    detector.close()

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    stability = _compute_mask_stability(masks)

    return {
        "mode": "Person-aware HSV",
        "avg_fps": fps,
        "avg_latency_ms": avg_latency,
        "stability": stability,
        "mask_count": len(masks),
    }


def benchmark_ai_hybrid(
    frames: list[np.ndarray],
    detection_cfg: DetectionConfig,
    processing_cfg: ProcessingConfig,
    ai_cfg: AIConfig,
) -> dict:
    """Benchmark AI hybrid detection (ONNX + HSV)."""
    try:
        detector = AIHybridDetector(detection_cfg, processing_cfg, ai_cfg)
    except Exception as e:
        return {
            "mode": "AI Hybrid",
            "avg_fps": 0.0,
            "avg_latency_ms": 0.0,
            "stability": 0.0,
            "mask_count": 0,
            "error": str(e),
        }

    masks: list[np.ndarray] = []
    latencies: list[float] = []
    ai_latencies: list[float] = []

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        mask, _ = detector.detect(frame, timestamp_ms=i * 33)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ai_latencies.append(detector.last_ai_latency_ms)
        masks.append(mask)

    detector.close()

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    avg_ai_latency = float(np.mean(ai_latencies)) if ai_latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    stability = _compute_mask_stability(masks)

    return {
        "mode": "AI Hybrid",
        "avg_fps": fps,
        "avg_latency_ms": avg_latency,
        "avg_ai_latency_ms": avg_ai_latency,
        "stability": stability,
        "mask_count": len(masks),
    }


def _print_results(results: list[dict]) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 72)
    print("DETECTION MODE BENCHMARK RESULTS")
    print("=" * 72)
    print(
        f"{'Mode':<22} {'FPS':>8} {'Latency':>10} {'AI Infer':>10} {'Stability':>10} {'Frames':>8}"
    )
    print("-" * 72)

    for r in results:
        error = r.get("error")
        if error:
            print(f"{r['mode']:<22} {'ERROR':>8} {error}")
            continue

        ai_str = f"{r.get('avg_ai_latency_ms', 0):.1f}ms" if "avg_ai_latency_ms" in r else "--"
        print(
            f"{r['mode']:<22} {r['avg_fps']:>7.1f}  "
            f"{r['avg_latency_ms']:>8.1f}ms  "
            f"{ai_str:>10}  "
            f"{r['stability']:>8.1%}  "
            f"{r['mask_count']:>7}"
        )

    print("=" * 72)
    print("\nNotes:")
    print("  - FPS: average frames per second (higher = better)")
    print("  - Latency: average detection time per frame (lower = better)")
    print("  - AI Infer: average AI model inference time (hybrid mode only)")
    print("  - Stability: fraction of frames with <5% mask change (higher = better)")
    print("  - These are ENGINEERING metrics, not accuracy metrics.")
    print("  - No ground truth is available; accuracy requires manual evaluation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark detection modes")
    parser.add_argument("--frames", type=int, default=100, help="Number of synthetic frames")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames (not counted)")
    args = parser.parse_args()

    print(
        f"Generating {args.frames + args.warmup} synthetic frames ({args.width}x{args.height})..."
    )
    all_frames = [
        _make_synthetic_frame(args.height, args.width, i) for i in range(args.frames + args.warmup)
    ]
    warmup_frames = all_frames[: args.warmup]
    bench_frames = all_frames[args.warmup :]

    # Default configs
    det_cfg = DetectionConfig(hsv_lower=[85, 100, 100], hsv_upper=[135, 255, 255])
    proc_cfg = ProcessingConfig(blur_kernel=1, morphology_kernel=1)
    mask_cfg = MaskConfig()
    ai_cfg = AIConfig(
        enabled=True,
        inference_width=320,
        inference_height=240,
        inference_frame_skip=1,
        confidence_threshold=0.5,
    )

    # Warmup HSV detector
    print("Warming up...")
    hsv_detector = BlueColorDetector(det_cfg, proc_cfg)
    for f in warmup_frames:
        hsv_detector.detect(f)

    results = []

    # Benchmark 1: HSV only
    print("Benchmarking HSV only...")
    results.append(benchmark_hsv(bench_frames, det_cfg, proc_cfg, mask_cfg))

    # Benchmark 2: Person-aware HSV (MediaPipe)
    print("Benchmarking Person-aware HSV...")
    results.append(benchmark_person_aware(bench_frames, det_cfg, proc_cfg, ai_cfg))

    # Benchmark 3: AI Hybrid (ONNX + HSV)
    print("Benchmarking AI Hybrid...")
    results.append(benchmark_ai_hybrid(bench_frames, det_cfg, proc_cfg, ai_cfg))

    _print_results(results)


if __name__ == "__main__":
    main()
