"""Tests for the help overlay UI component."""

from __future__ import annotations

import numpy as np

from cloak.ui.help_overlay import HelpOverlay


class TestHelpOverlayState:
    def test_initial_visible_is_false(self) -> None:
        overlay = HelpOverlay()
        assert overlay.visible is False

    def test_toggle_switches_visibility(self) -> None:
        overlay = HelpOverlay()
        overlay.toggle()
        assert overlay.visible is True
        overlay.toggle()
        assert overlay.visible is False

    def test_show_sets_visible_true(self) -> None:
        overlay = HelpOverlay()
        overlay.show()
        assert overlay.visible is True

    def test_hide_sets_visible_false(self) -> None:
        overlay = HelpOverlay()
        overlay.show()
        overlay.hide()
        assert overlay.visible is False


class TestHelpOverlayRender:
    def test_render_does_nothing_when_not_visible(self) -> None:
        overlay = HelpOverlay()
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        original = frame.copy()
        overlay.render(frame)
        np.testing.assert_array_equal(frame, original)

    def test_render_modifies_frame_when_visible(self) -> None:
        overlay = HelpOverlay()
        overlay.show()
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        original = frame.copy()
        overlay.render(frame)
        assert not np.array_equal(frame, original)

    def test_render_handles_different_frame_sizes(self) -> None:
        overlay = HelpOverlay()
        overlay.show()
        for h, w in [(100, 100), (480, 640), (720, 1280)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            overlay.render(frame)
            assert frame.shape == (h, w, 3)
