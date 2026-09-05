"""Tests for performance tracking using synthetic timing."""

from __future__ import annotations

import time

from cloak.monitoring.performance import PerformanceTracker


class TestStartStopCycle:
    """Start/stop produces a positive elapsed time."""

    def test_single_stage(self):
        tracker = PerformanceTracker()
        tracker.start("detect")
        time.sleep(0.001)  # 1ms
        tracker.stop("detect")
        stats = tracker.get_stats()
        assert stats["detect"]["avg_ms"] > 0

    def test_multiple_stages(self):
        tracker = PerformanceTracker()
        for stage in ("detect", "refine", "render"):
            tracker.start(stage)
            time.sleep(0.001)
            tracker.stop(stage)
        stats = tracker.get_stats()
        for stage in ("detect", "refine", "render"):
            assert stats[stage]["avg_ms"] > 0


class TestRollingWindow:
    """Old entries are discarded after WINDOW_SIZE frames."""

    def test_window_size(self):
        tracker = PerformanceTracker()
        for _ in range(PerformanceTracker.WINDOW_SIZE + 10):
            tracker.start("detect")
            time.sleep(0.0001)
            tracker.stop("detect")
        assert len(tracker._timings["detect"]) == PerformanceTracker.WINDOW_SIZE


class TestGetStats:
    """get_stats returns all stage names."""

    def test_all_stages_present(self):
        tracker = PerformanceTracker()
        stats = tracker.get_stats()
        for stage in PerformanceTracker.STAGES:
            assert stage in stats
        assert "total" in stats
        assert "fps" in stats


class TestReset:
    """Reset clears all accumulated data."""

    def test_reset_clears(self):
        tracker = PerformanceTracker()
        tracker.start("detect")
        time.sleep(0.001)
        tracker.stop("detect")
        tracker.start_frame()
        time.sleep(0.001)
        tracker.stop_frame()
        tracker.reset()
        stats = tracker.get_stats()
        assert stats["detect"]["avg_ms"] == 0.0
        assert stats["total"]["avg_ms"] == 0.0


class TestFrameTiming:
    """start_frame/stop_frame records total frame time."""

    def test_frame_time_positive(self):
        tracker = PerformanceTracker()
        tracker.start_frame()
        time.sleep(0.001)
        tracker.stop_frame()
        stats = tracker.get_stats()
        assert stats["total"]["avg_ms"] > 0

    def test_fps_positive(self):
        tracker = PerformanceTracker()
        tracker.start_frame()
        time.sleep(0.01)
        tracker.stop_frame()
        stats = tracker.get_stats()
        assert stats["fps"]["avg_ms"] > 0
