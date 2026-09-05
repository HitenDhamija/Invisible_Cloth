"""Per-stage performance measurement for the frame pipeline.

Tracks timing for each major stage using time.perf_counter()
and maintains a rolling window of the last 30 frames.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class PerformanceTracker:
    """Measure and report per-stage frame processing times.

    Example::

        tracker = PerformanceTracker()
        tracker.start("detect")
        raw_mask, stats = detector.detect(frame)
        tracker.stop("detect")

        # After many frames:
        stats = tracker.get_stats()
        print(stats["detect"]["avg_ms"])
    """

    STAGES = ("capture", "preprocess", "detect", "refine", "temporal", "render")
    WINDOW_SIZE = 30

    def __init__(self) -> None:
        self._timings: dict[str, deque[float]] = {
            stage: deque(maxlen=self.WINDOW_SIZE) for stage in self.STAGES
        }
        self._start_times: dict[str, float] = {}
        self._frame_times: deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._total_start: float | None = None

    def start_frame(self) -> None:
        """Mark the beginning of a new frame."""
        self._total_start = time.perf_counter()

    def start(self, stage: str) -> None:
        """Record the start time for a pipeline stage."""
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str) -> None:
        """Record the end time for a pipeline stage and store the delta."""
        if stage not in self._start_times:
            return
        elapsed = time.perf_counter() - self._start_times.pop(stage)
        self._timings[stage].append(elapsed)

    def stop_frame(self) -> None:
        """Mark the end of frame processing and record total frame time."""
        if self._total_start is not None:
            self._frame_times.append(time.perf_counter() - self._total_start)
            self._total_start = None

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return averaged stats for all stages plus totals.

        Returns:
            Dict mapping stage name to {"avg_ms", "min_ms", "max_ms"}.
            Includes "total" and "fps" keys for the full frame.
        """
        stats: dict[str, dict[str, float]] = {}

        for stage in self.STAGES:
            times = self._timings[stage]
            if times:
                arr = np.array(times) * 1000.0  # seconds -> ms
                stats[stage] = {
                    "avg_ms": float(np.mean(arr)),
                    "min_ms": float(np.min(arr)),
                    "max_ms": float(np.max(arr)),
                }
            else:
                stats[stage] = {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

        if self._frame_times:
            ft = np.array(self._frame_times) * 1000.0
            stats["total"] = {
                "avg_ms": float(np.mean(ft)),
                "min_ms": float(np.min(ft)),
                "max_ms": float(np.max(ft)),
            }
            stats["fps"] = {"avg_ms": 1000.0 / float(np.mean(ft))}
        else:
            stats["total"] = {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
            stats["fps"] = {"avg_ms": 0.0}

        return stats

    def reset(self) -> None:
        """Clear all accumulated timing data."""
        for stage in self.STAGES:
            self._timings[stage].clear()
        self._frame_times.clear()
        self._start_times.clear()
        self._total_start = None
