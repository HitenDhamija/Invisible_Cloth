"""Tests for blue-cloth detection using synthetic images (no webcam required)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cloak.config.schemas import DetectionConfig, ProcessingConfig
from cloak.detection.detector import BlueColorDetector

# -- helpers -------------------------------------------------------------------


def _make_bgr(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    """Create a solid BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _make_blue_rect(
    h: int,
    w: int,
    background_bgr: tuple[int, int, int],
    rect_bgr: tuple[int, int, int],
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Create a frame with a rectangle of *rect_bgr* inside *background_bgr*."""
    frame = _make_bgr(h, w, background_bgr)
    frame[y0:y1, x0:x1] = rect_bgr
    return frame


def _bgr_to_hsv(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert a BGR tuple to HSV (approximate, for reference)."""
    px = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)
    return tuple(int(v) for v in hsv[0, 0])  # type: ignore[return-value]


# Well-known blue in BGR: pure blue is (255, 0, 0)
# In HSV this is roughly H=120, S=255, V=255
_BLUE_BGR = (255, 0, 0)
_RED_BGR = (0, 0, 255)
_GREEN_BGR = (0, 255, 0)
_GREY_BGR = (128, 128, 128)


# -- default config fixture ---------------------------------------------------


@pytest.fixture
def detection_cfg() -> DetectionConfig:
    return DetectionConfig(
        hsv_lower=[85, 100, 100],
        hsv_upper=[135, 255, 255],
    )


@pytest.fixture
def processing_cfg() -> ProcessingConfig:
    return ProcessingConfig(blur_kernel=1, morphology_kernel=1)


@pytest.fixture
def detector(detection_cfg: DetectionConfig, processing_cfg: ProcessingConfig) -> BlueColorDetector:
    return BlueColorDetector(detection_cfg, processing_cfg)


# -- basic detection tests ----------------------------------------------------


class TestBlueDetection:
    def test_solid_blue_frame(self, detector: BlueColorDetector) -> None:
        """A frame entirely blue should produce an all-white mask."""
        frame = _make_bgr(100, 100, _BLUE_BGR)
        mask, stats = detector.detect(frame)
        assert mask.shape == (100, 100)
        assert stats.cloak_ratio == pytest.approx(1.0, abs=0.01)

    def test_no_blue_frame(self, detector: BlueColorDetector) -> None:
        """A red frame should produce an all-black mask."""
        frame = _make_bgr(100, 100, _RED_BGR)
        mask, stats = detector.detect(frame)
        assert stats.cloak_ratio == pytest.approx(0.0, abs=0.01)

    def test_grey_frame(self, detector: BlueColorDetector) -> None:
        """Grey (low saturation) should NOT be detected as blue."""
        frame = _make_bgr(100, 100, _GREY_BGR)
        mask, stats = detector.detect(frame)
        assert stats.cloak_ratio == pytest.approx(0.0, abs=0.01)

    def test_green_frame(self, detector: BlueColorDetector) -> None:
        """Green should not be detected as blue."""
        frame = _make_bgr(100, 100, _GREEN_BGR)
        mask, stats = detector.detect(frame)
        assert stats.cloak_ratio == pytest.approx(0.0, abs=0.01)


# -- spatial tests (blue rectangle in non-blue background) --------------------


class TestSpatialDetection:
    def test_blue_rect_in_red_frame(self, detector: BlueColorDetector) -> None:
        """Detect a 50x50 blue rectangle in a 100x100 red frame."""
        frame = _make_blue_rect(100, 100, _RED_BGR, _BLUE_BGR, 25, 75, 25, 75)
        mask, stats = detector.detect(frame)

        # The blue rectangle is 50x50 = 2500 out of 10000 pixels
        assert stats.cloak_pixels >= 2400  # allow some boundary erosion
        assert stats.cloak_ratio == pytest.approx(0.25, abs=0.05)

        # The mask inside the rectangle should be white
        inner_mask = mask[35:65, 35:65]
        assert np.all(inner_mask == 255)

        # The mask in the far corner should be black
        assert mask[5, 5] == 0

    def test_half_blue_frame(self, detector: BlueColorDetector) -> None:
        """Left half blue, right half red → ~50% detection."""
        frame = _make_bgr(100, 100, _RED_BGR)
        frame[:, :50] = _BLUE_BGR
        mask, stats = detector.detect(frame)
        assert stats.cloak_ratio == pytest.approx(0.5, abs=0.05)


# -- mask is binary -----------------------------------------------------------


class TestMaskProperties:
    def test_mask_is_binary(self, detector: BlueColorDetector) -> None:
        frame = _make_blue_rect(80, 120, _GREY_BGR, _BLUE_BGR, 10, 70, 10, 110)
        mask, _ = detector.detect(frame)
        unique = np.unique(mask)
        assert set(unique.tolist()).issubset({0, 255})

    def test_mask_dtype_is_uint8(self, detector: BlueColorDetector) -> None:
        frame = _make_bgr(50, 50, _BLUE_BGR)
        mask, _ = detector.detect(frame)
        assert mask.dtype == np.uint8


# -- statistics ----------------------------------------------------------------


class TestDetectionStats:
    def test_total_pixels(self, detector: BlueColorDetector) -> None:
        frame = _make_bgr(120, 80, _BLUE_BGR)
        _, stats = detector.detect(frame)
        assert stats.total_pixels == 120 * 80

    def test_cloak_pixels_match_mask(self, detector: BlueColorDetector) -> None:
        frame = _make_bgr(60, 60, _BLUE_BGR)
        mask, stats = detector.detect(frame)
        assert stats.cloak_pixels == cv2.countNonZero(mask)

    def test_warning_when_too_broad(self) -> None:
        """If nearly the entire image is blue, a warning should appear."""
        cfg = DetectionConfig(hsv_lower=[0, 0, 0], hsv_upper=[179, 255, 255])
        proc = ProcessingConfig(blur_kernel=1, morphology_kernel=1)
        det = BlueColorDetector(cfg, proc)

        # Black frame — everything matches the wide range
        frame = _make_bgr(100, 100, (0, 0, 0))
        _, stats = det.detect(frame)
        assert stats.warning is not None
        assert "too broad" in stats.warning


# -- set_bounds (runtime update) -----------------------------------------------


class TestSetBounds:
    def test_update_bounds_changes_detection(self) -> None:
        cfg = DetectionConfig(hsv_lower=[85, 100, 100], hsv_upper=[135, 255, 255])
        proc = ProcessingConfig(blur_kernel=1, morphology_kernel=1)
        det = BlueColorDetector(cfg, proc)

        frame = _make_bgr(50, 50, _BLUE_BGR)

        # With blue range, blue is detected
        _, stats1 = det.detect(frame)
        assert stats1.cloak_ratio > 0.9

        # Shift to red range — blue should disappear
        det.set_bounds([0, 100, 100], [10, 255, 255])
        _, stats2 = det.detect(frame)
        assert stats2.cloak_ratio < 0.01


# -- detect_blue_region --------------------------------------------------------


class TestDetectBlueRegion:
    def test_non_blue_becomes_black(self, detector: BlueColorDetector) -> None:
        frame = _make_blue_rect(80, 80, _RED_BGR, _BLUE_BGR, 20, 60, 20, 60)
        mask, _ = detector.detect(frame)
        region = detector.detect_blue_region(frame, mask)

        # Corners should be black
        assert np.all(region[0, 0] == 0)
        assert np.all(region[79, 79] == 0)

        # Center of blue rect should be preserved
        center = region[40, 40]
        assert np.all(center == _BLUE_BGR)


# -- morphological cleanup ----------------------------------------------------


class TestMorphology:
    def test_smoothing_reduces_noise(self) -> None:
        """With blur enabled, isolated single-pixel noise may be smoothed out."""
        cfg = DetectionConfig(hsv_lower=[85, 100, 100], hsv_upper=[135, 255, 255])
        proc = ProcessingConfig(blur_kernel=5, morphology_kernel=5)
        det = BlueColorDetector(cfg, proc)

        # Frame with lots of noise — single blue pixels scattered
        frame = _make_bgr(100, 100, _RED_BGR)
        rng = np.random.default_rng(42)
        noise_y = rng.integers(0, 100, size=50)
        noise_x = rng.integers(0, 100, size=50)
        frame[noise_y, noise_x] = _BLUE_BGR

        mask, stats = det.detect(frame)
        # Morphological opening should remove most single-pixel noise
        assert stats.cloak_ratio < 0.05


# -- edge cases ----------------------------------------------------------------


class TestEdgeCases:
    def test_single_pixel_frame(self, detector: BlueColorDetector) -> None:
        frame = np.array([[[255, 0, 0]]], dtype=np.uint8)
        mask, stats = detector.detect(frame)
        assert mask.shape == (1, 1)
        assert stats.total_pixels == 1

    def test_very_small_frame(self, detector: BlueColorDetector) -> None:
        frame = _make_bgr(2, 2, _BLUE_BGR)
        mask, stats = detector.detect(frame)
        assert stats.cloak_pixels == 4
