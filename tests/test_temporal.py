"""Tests for temporal mask smoothing using synthetic mask sequences."""

from __future__ import annotations

import numpy as np

from cloak.config.schemas import TemporalConfig
from cloak.processing.temporal import TemporalMaskSmoother

# -- helpers -------------------------------------------------------------------


def _empty_mask(h: int = 100, w: int = 100) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _full_mask(h: int = 100, w: int = 100) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _rect_mask(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = _empty_mask(h, w)
    mask[y0:y1, x0:x1] = 255
    return mask


# -- EMA alpha=1.0 tests ------------------------------------------------------


class TestEMAAlphaOne:
    """Alpha=1.0 means no smoothing -- output matches input."""

    def test_output_matches_input(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=1.0, persistence_frames=0)
        smoother = TemporalMaskSmoother(cfg)
        mask = _rect_mask(100, 100, 20, 80, 20, 80)
        result = smoother.smooth(mask)
        np.testing.assert_array_equal(result, mask)

    def test_sequential_frames_match_input(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=1.0, persistence_frames=0)
        smoother = TemporalMaskSmoother(cfg)
        for _ in range(5):
            mask = _full_mask()
            result = smoother.smooth(mask)
            np.testing.assert_array_equal(result, mask)


# -- EMA alpha=0.0 tests ------------------------------------------------------


class TestEMAAlphaZero:
    """Alpha=0.0 means frozen -- output stays at initial state."""

    def test_output_frozen_after_first_frame(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=0.0, persistence_frames=0)
        smoother = TemporalMaskSmoother(cfg)
        smoother.smooth(_full_mask())
        # With alpha=0, accumulated stays at 255, threshold=0, so output stays 255
        result = smoother.smooth(_empty_mask())
        np.testing.assert_array_equal(result, _full_mask())


# -- persistence tests --------------------------------------------------------


class TestPersistence:
    """Persistence counters keep pixels ON after they disappear."""

    def test_persistence_keeps_pixel_on(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=1.0, persistence_frames=3)
        smoother = TemporalMaskSmoother(cfg)

        # Frame 1: pixel ON
        mask1 = _rect_mask(10, 10, 0, 5, 0, 5)
        smoother.smooth(mask1)

        # Frame 2: pixel OFF -- persistence should keep it ON
        mask2 = _empty_mask(10, 10)
        result = smoother.smooth(mask2)
        assert np.sum(result == 255) > 0, "Persistence should keep pixels ON"

    def test_persistence_expires(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=1.0, persistence_frames=2)
        smoother = TemporalMaskSmoother(cfg)

        # Frame 1: pixel ON
        smoother.smooth(_rect_mask(10, 10, 0, 5, 0, 5))

        # Frames 2-5: pixel OFF -- persistence expires after 2 frames
        for _ in range(4):
            result = smoother.smooth(_empty_mask(10, 10))

        # After 4 frames with persistence=2, should be OFF
        assert np.sum(result == 255) == 0, "Persistence should expire"


# -- reset tests --------------------------------------------------------------


class TestReset:
    """Reset clears all accumulated state."""

    def test_reset_clears_state(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=0.5, persistence_frames=3)
        smoother = TemporalMaskSmoother(cfg)

        smoother.smooth(_full_mask())
        smoother.smooth(_full_mask())
        smoother.reset()

        assert smoother.frame_count == 0
        # After reset, next frame starts fresh
        result = smoother.smooth(_empty_mask(100, 100))
        np.testing.assert_array_equal(result, _empty_mask(100, 100))


# -- alternating masks --------------------------------------------------------


class TestAlternatingMasks:
    """EMA smoothing delays transitions compared to raw input."""

    def test_single_off_frame_doesnt_flip_output(self):
        # With alpha=0.5, a single OFF frame after many ON frames
        # should not immediately flip the output to OFF
        cfg = TemporalConfig(enabled=True, ema_alpha=0.5, persistence_frames=0)
        smoother = TemporalMaskSmoother(cfg)

        # 5 full frames to build up accumulated
        for _ in range(5):
            smoother.smooth(_full_mask())

        # 1 empty frame
        result = smoother.smooth(_empty_mask())
        # EMA should keep it ON (accumulated is still above threshold)
        assert np.sum(result == 255) > 0, "Single OFF frame should not flip output"

    def test_many_off_frames_eventually_turn_off(self):
        # After many consecutive OFF frames, output should eventually turn OFF
        cfg = TemporalConfig(enabled=True, ema_alpha=0.5, persistence_frames=0)
        smoother = TemporalMaskSmoother(cfg)

        # Build up with full masks
        for _ in range(3):
            smoother.smooth(_full_mask())

        # Feed many empty frames
        for _ in range(20):
            result = smoother.smooth(_empty_mask())

        assert np.sum(result == 255) == 0, "Many OFF frames should eventually turn output OFF"


# -- edge cases ---------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty masks, full masks, single pixels."""

    def test_empty_mask_stays_empty(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=0.5, persistence_frames=0)
        smoother = TemporalMaskSmoother(cfg)
        for _ in range(5):
            result = smoother.smooth(_empty_mask())
        np.testing.assert_array_equal(result, _empty_mask())

    def test_full_mask_stays_full(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=0.5, persistence_frames=0)
        smoother = TemporalMaskSmoother(cfg)
        for _ in range(5):
            result = smoother.smooth(_full_mask())
        np.testing.assert_array_equal(result, _full_mask())

    def test_1x1_mask(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=0.6, persistence_frames=1)
        smoother = TemporalMaskSmoother(cfg)
        mask = np.array([[255]], dtype=np.uint8)
        result = smoother.smooth(mask)
        assert result.shape == (1, 1)
        assert result.dtype == np.uint8

    def test_output_is_uint8(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=0.6, persistence_frames=2)
        smoother = TemporalMaskSmoother(cfg)
        result = smoother.smooth(_full_mask())
        assert result.dtype == np.uint8

    def test_output_is_binary(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=0.6, persistence_frames=2)
        smoother = TemporalMaskSmoother(cfg)
        result = smoother.smooth(_full_mask())
        unique = np.unique(result)
        assert all(v in (0, 255) for v in unique)

    def test_shape_preserved(self):
        cfg = TemporalConfig(enabled=True, ema_alpha=0.6, persistence_frames=1)
        smoother = TemporalMaskSmoother(cfg)
        mask = _rect_mask(120, 160, 10, 50, 10, 80)
        result = smoother.smooth(mask)
        assert result.shape == mask.shape

    def test_disabled_returns_input(self):
        cfg = TemporalConfig(enabled=False)
        smoother = TemporalMaskSmoother(cfg)
        mask = _rect_mask(100, 100, 0, 50, 0, 50)
        result = smoother.smooth(mask)
        np.testing.assert_array_equal(result, mask)
