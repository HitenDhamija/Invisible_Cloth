"""Tests for the error display UI component."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from cloak.ui.error_display import ErrorDisplay


class TestErrorDisplayState:
    def test_initial_active_is_false(self) -> None:
        err = ErrorDisplay()
        assert err.active is False

    def test_initial_message_is_none(self) -> None:
        err = ErrorDisplay()
        assert err.message is None

    def test_show_sets_active_and_message(self) -> None:
        err = ErrorDisplay()
        err.show("test error")
        assert err.active is True
        assert err.message == "test error"

    def test_clear_resets_state(self) -> None:
        err = ErrorDisplay()
        err.show("error")
        err.clear()
        assert err.active is False
        assert err.message is None


class TestErrorDisplayRender:
    def test_render_does_nothing_when_not_active(self) -> None:
        err = ErrorDisplay()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        original = frame.copy()
        err.render(frame)
        np.testing.assert_array_equal(frame, original)

    def test_render_draws_error_bar_when_active(self) -> None:
        err = ErrorDisplay()
        err.show("error message")
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        original = frame.copy()
        with patch("cloak.ui.error_display.time.perf_counter", return_value=0.0):
            err.render(frame)
        assert not np.array_equal(frame, original)

    def test_auto_hide_after_duration(self) -> None:
        err = ErrorDisplay(duration_seconds=1.0)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch("cloak.ui.error_display.time.perf_counter", return_value=0.0):
            err.show("error")
            err.render(frame)
        assert err.active is True
        with patch("cloak.ui.error_display.time.perf_counter", return_value=1.5):
            err.render(frame)
        assert err.active is False

    def test_custom_duration_parameter(self) -> None:
        err = ErrorDisplay()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch("cloak.ui.error_display.time.perf_counter", return_value=0.0):
            err.show("error", duration=0.5)
            err.render(frame)
        assert err.active is True
        with patch("cloak.ui.error_display.time.perf_counter", return_value=0.6):
            err.render(frame)
        assert err.active is False

    def test_render_handles_different_frame_sizes(self) -> None:
        err = ErrorDisplay()
        err.show("error")
        for h, w in [(50, 100), (240, 320), (480, 640)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            with patch("cloak.ui.error_display.time.perf_counter", return_value=0.0):
                err.render(frame)
            assert frame.shape == (h, w, 3)
