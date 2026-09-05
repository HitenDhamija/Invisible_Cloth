"""Tests for adaptive illumination preprocessing using synthetic images."""

from __future__ import annotations

import cv2
import numpy as np

from cloak.config.schemas import AdaptiveConfig
from cloak.detection.adaptive import AdaptivePreprocessor

# -- helpers -------------------------------------------------------------------


def _dark_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Very dark BGR frame (low V channel)."""
    return np.full((h, w, 3), 20, dtype=np.uint8)


def _bright_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Very bright BGR frame (high V channel)."""
    return np.full((h, w, 3), 240, dtype=np.uint8)


def _normal_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Mid-range BGR frame."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


# -- brightness normalization tests -------------------------------------------


class TestBrightnessNormalization:
    """Brightness normalization scales V channel toward mean=128."""

    def test_dark_frame_brightened(self):
        cfg = AdaptiveConfig(
            enabled=True,
            brightness_normalize=True,
            clahe_clip=0.5,
            clahe_grid=8,
        )
        preprocessor = AdaptivePreprocessor(cfg)
        dark = _dark_frame()
        result = preprocessor.preprocess(dark)
        v_orig = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
        v_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
        assert abs(v_result - 128) < abs(v_orig - 128), "Dark frame should be brightened"

    def test_bright_frame_darkened(self):
        cfg = AdaptiveConfig(
            enabled=True,
            brightness_normalize=True,
            clahe_clip=0.5,
            clahe_grid=8,
        )
        preprocessor = AdaptivePreprocessor(cfg)
        bright = _bright_frame()
        result = preprocessor.preprocess(bright)
        v_orig = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
        v_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
        assert abs(v_result - 128) < abs(v_orig - 128), "Bright frame should be darkened"


# -- CLAHE tests --------------------------------------------------------------


class TestCLAHE:
    """CLAHE modifies the V channel."""

    def test_clahe_changes_output(self):
        cfg = AdaptiveConfig(
            enabled=True,
            brightness_normalize=False,
            clahe_clip=2.0,
            clahe_grid=8,
        )
        preprocessor = AdaptivePreprocessor(cfg)
        frame = _normal_frame()
        result = preprocessor.preprocess(frame)
        assert not np.array_equal(result, frame), "CLAHE should modify the frame"


# -- disabled passthrough tests -----------------------------------------------


class TestDisabledPassthrough:
    """When disabled, output equals input."""

    def test_passthrough(self):
        cfg = AdaptiveConfig(enabled=False)
        preprocessor = AdaptivePreprocessor(cfg)
        frame = _normal_frame()
        result = preprocessor.preprocess(frame)
        np.testing.assert_array_equal(result, frame)


# -- shape preservation tests -------------------------------------------------


class TestShapePreservation:
    """Output shape matches input shape."""

    def test_shape(self):
        cfg = AdaptiveConfig(enabled=True, clahe_clip=2.0, clahe_grid=8)
        preprocessor = AdaptivePreprocessor(cfg)
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        result = preprocessor.preprocess(frame)
        assert result.shape == frame.shape
