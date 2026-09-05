"""Tests for automatic HSV calibration using synthetic data."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from cloak.config.schemas import CalibrationConfig, DetectionConfig
from cloak.detection.auto_calibrator import AutoCalibrator, CalibrationResult, CalibrationState

# -- helpers -------------------------------------------------------------------


def _make_bgr(h: int = 100, w: int = 100, bgr: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _make_blue_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a frame that is entirely blue (simulates full-ROI cloak)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Blue BGR: (255, 50, 30) — fills the entire frame
    frame[:, :] = (255, 50, 30)
    return frame


def _make_blue_rect_frame(
    h: int = 100, w: int = 100,
    fill_fraction: float = 0.9,
) -> np.ndarray:
    """Create a frame with blue filling most of the ROI area."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    margin = int(h * (1 - fill_fraction) / 2)
    frame[margin : h - margin, margin : w - margin] = (255, 50, 30)
    return frame


def _make_frame_with_outliers(
    h: int = 100, w: int = 100,
    main_bgr: tuple[int, int, int] = (255, 50, 30),
    outlier_bgr: tuple[int, int, int] = (0, 0, 255),
    outlier_fraction: float = 0.02,
) -> np.ndarray:
    """Create a frame with a main color and scattered outliers."""
    frame = np.full((h, w, 3), main_bgr, dtype=np.uint8)
    # Scatter outliers
    n_outliers = int(h * w * outlier_fraction)
    rng = np.random.default_rng(42)
    rows = rng.integers(0, h, size=n_outliers)
    cols = rng.integers(0, w, size=n_outliers)
    frame[rows, cols] = outlier_bgr
    return frame


def _default_cal_config(**kwargs) -> CalibrationConfig:
    defaults = dict(
        roi_fraction=0.5,  # large ROI for testing
        min_pixels=50,
        percentile_low=2.0,
        percentile_high=98.0,
        h_margin=5,
        s_margin=10,
        v_margin=10,
        use_kmeans=False,
        use_histogram=False,
    )
    defaults.update(kwargs)
    return CalibrationConfig(**defaults)


def _default_detection_config(**kwargs) -> DetectionConfig:
    return DetectionConfig(**kwargs)


# -- state machine tests ------------------------------------------------------


class TestCalibrationStateMachine:
    """Test state transitions."""

    def test_initial_state(self):
        cal_cfg = _default_cal_config()
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        assert cal.state == CalibrationState.INACTIVE
        assert cal.is_active is False

    def test_start_transitions_to_collecting(self):
        cal_cfg = _default_cal_config()
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        cal.start((480, 640, 3))
        assert cal.state == CalibrationState.COLLECTING
        assert cal.is_active is True

    def test_cancel_returns_to_inactive(self):
        cal_cfg = _default_cal_config()
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        cal.start((480, 640, 3))
        cal.cancel()
        assert cal.state == CalibrationState.INACTIVE
        assert cal.is_active is False


# -- pixel collection tests ---------------------------------------------------


class TestPixelCollection:
    """Test pixel collection and computation."""

    def test_collect_returns_result(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        result = cal.collect(frame)

        assert result is not None
        assert isinstance(result, CalibrationResult)
        assert result.pixel_count > 0
        assert len(result.hsv_lower) == 3
        assert len(result.hsv_upper) == 3

    def test_collect_enters_preview_state(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        cal.collect(frame)
        assert cal.state == CalibrationState.PREVIEW

    def test_insufficient_pixels_returns_none(self):
        cal_cfg = _default_cal_config(min_pixels=10000)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(10, 10)  # too small
        cal.start(frame.shape)
        result = cal.collect(frame)
        assert result is None
        assert cal.state == CalibrationState.COLLECTING

    def test_accept_returns_bounds(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        cal.collect(frame)
        lower, upper = cal.accept()

        assert len(lower) == 3
        assert len(upper) == 3
        assert cal.state == CalibrationState.INACTIVE

    def test_accept_without_result_raises(self):
        cal_cfg = _default_cal_config()
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        with pytest.raises(ValueError, match="No calibration result"):
            cal.accept()


# -- percentile algorithm tests ------------------------------------------------


class TestPercentileAlgorithm:
    """Test the core percentile-based calibration algorithm."""

    def test_blue_frame_produces_valid_bounds(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        result = cal.collect(frame)

        # Blue in HSV is around H=100-130
        assert result.hsv_lower[0] >= 70
        assert result.hsv_upper[0] <= 150
        # Saturation should be high
        assert result.hsv_lower[1] >= 50
        # Value should be reasonable
        assert result.hsv_lower[2] >= 20

    def test_outliers_do_not_shift_bounds(self):
        """Percentile statistics should be robust to outliers."""
        cal_cfg = _default_cal_config(roi_fraction=0.8, percentile_low=5.0, percentile_high=95.0)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)

        # Frame with blue center and red outliers
        frame = _make_frame_with_outliers(
            h=100, w=100,
            main_bgr=(255, 50, 30),  # blue
            outlier_bgr=(0, 0, 255),  # red
            outlier_fraction=0.02,
        )
        cal.start(frame.shape)
        result = cal.collect(frame)

        # The bounds should still primarily capture blue
        assert result.hsv_lower[0] >= 70
        assert result.hsv_upper[0] <= 150

    def test_method_is_percentile(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8, use_histogram=False, use_kmeans=False)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        result = cal.collect(frame)
        assert result.method == "percentile"

    def test_median_hsv_is_computed(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        result = cal.collect(frame)

        assert len(result.median_hsv) == 3
        assert all(isinstance(v, float) for v in result.median_hsv)

    def test_iqr_is_computed(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        result = cal.collect(frame)

        assert len(result.iqr_hsv) == 3
        assert all(isinstance(v, float) for v in result.iqr_hsv)


# -- histogram refinement tests ------------------------------------------------


class TestHistogramRefinement:
    """Test histogram-based refinement."""

    def test_histogram_method_label(self):
        cal_cfg = _default_cal_config(
            roi_fraction=0.8, use_histogram=True, use_kmeans=False,
        )
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        result = cal.collect(frame)
        assert "histogram" in result.method

    def test_histogram_tightens_bounds(self):
        """Histogram refinement should produce equal or tighter bounds."""
        # Without histogram
        cal_cfg_no_hist = _default_cal_config(
            roi_fraction=0.8, use_histogram=False,
        )
        det_cfg = _default_detection_config()
        cal_no_hist = AutoCalibrator(cal_cfg_no_hist, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal_no_hist.start(frame.shape)
        r1 = cal_no_hist.collect(frame)

        # With histogram
        cal_cfg_hist = _default_cal_config(
            roi_fraction=0.8, use_histogram=True,
        )
        cal_hist = AutoCalibrator(cal_cfg_hist, det_cfg)
        cal_hist.start(frame.shape)
        r2 = cal_hist.collect(frame)

        # Histogram bounds should be equal or tighter
        for ch in range(3):
            assert r2.hsv_lower[ch] >= r1.hsv_lower[ch] - 1
            assert r2.hsv_upper[ch] <= r1.hsv_upper[ch] + 1


# -- K-means tests ------------------------------------------------------------


class TestKMeansRefinement:
    """Test K-means clustering refinement."""

    def test_kmeans_method_label(self):
        cal_cfg = _default_cal_config(
            roi_fraction=0.8, use_kmeans=True, use_histogram=False,
        )
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        result = cal.collect(frame)
        assert "kmeans" in result.method

    def test_kmeans_single_cluster_comparable(self):
        """With varied color, K-means and percentile should produce valid bounds."""
        cal_cfg_perc = _default_cal_config(
            roi_fraction=0.8, use_kmeans=False, use_histogram=False,
        )
        det_cfg = _default_detection_config()
        cal_perc = AutoCalibrator(cal_cfg_perc, det_cfg)
        # Use a frame with slight color variation (more realistic)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        # Blue with slight HSV variation
        for y in range(100):
            for x in range(100):
                h_val = 117 + rng.integers(-3, 4)
                s_val = 225 + rng.integers(-10, 11)
                v_val = 255 + rng.integers(-20, 21)
                frame[y, x] = cv2.cvtColor(
                    np.uint8([[[np.clip(h_val, 0, 179),
                                np.clip(s_val, 0, 255),
                                np.clip(v_val, 0, 255)]]]),
                    cv2.COLOR_HSV2BGR,
                )[0, 0]
        cal_perc.start(frame.shape)
        r1 = cal_perc.collect(frame)

        cal_cfg_km = _default_cal_config(
            roi_fraction=0.8, use_kmeans=True, use_histogram=False,
        )
        cal_km = AutoCalibrator(cal_cfg_km, det_cfg)
        cal_km.start(frame.shape)
        r2 = cal_km.collect(frame)

        # Both should produce valid bounds (H in blue range)
        assert r1.hsv_lower[0] >= 70
        assert r1.hsv_upper[0] <= 150
        assert r2.hsv_lower[0] >= 70
        assert r2.hsv_upper[0] <= 150


# -- ROI tests -----------------------------------------------------------------


class TestROI:
    """Test ROI computation and pixel extraction."""

    def test_roi_is_centered(self):
        cal_cfg = _default_cal_config(roi_fraction=0.5)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        cal.start((480, 640, 3))

        x0, y0, rw, rh = cal._roi_rect
        # ROI should be roughly centered
        assert abs(x0 - (640 - rw) // 2) < 2
        assert abs(y0 - (480 - rh) // 2) < 2

    def test_roi_fraction_affects_size(self):
        cal_cfg_small = _default_cal_config(roi_fraction=0.2)
        det_cfg = _default_detection_config()
        cal_small = AutoCalibrator(cal_cfg_small, det_cfg)
        cal_small.start((480, 640, 3))
        _, _, rw_small, rh_small = cal_small._roi_rect

        cal_cfg_large = _default_cal_config(roi_fraction=0.6)
        cal_large = AutoCalibrator(cal_cfg_large, det_cfg)
        cal_large.start((480, 640, 3))
        _, _, rw_large, rh_large = cal_large._roi_rect

        assert rw_large > rw_small
        assert rh_large > rh_small


# -- preview mask tests --------------------------------------------------------


class TestPreviewMask:
    """Test preview mask generation."""

    def test_preview_mask_generated_after_collect(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        cal.collect(frame)

        assert cal.preview_mask is not None
        assert cal.preview_mask.shape == (100, 100)
        assert cal.preview_mask.dtype == np.uint8

    def test_preview_mask_has_blue_pixels(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        cal.collect(frame)

        assert np.count_nonzero(cal.preview_mask) > 0

    def test_preview_mask_none_before_collect(self):
        cal_cfg = _default_cal_config()
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        assert cal.preview_mask is None


# -- config schema tests -------------------------------------------------------


class TestCalibrationConfigDefaults:
    """CalibrationConfig has correct defaults."""

    def test_default_roi_fraction(self):
        cfg = CalibrationConfig()
        assert cfg.roi_fraction == 0.25

    def test_default_percentiles(self):
        cfg = CalibrationConfig()
        assert cfg.percentile_low == 2.0
        assert cfg.percentile_high == 98.0

    def test_default_margins(self):
        cfg = CalibrationConfig()
        assert cfg.h_margin == 8
        assert cfg.s_margin == 15
        assert cfg.v_margin == 15

    def test_default_kmeans_disabled(self):
        cfg = CalibrationConfig()
        assert cfg.use_kmeans is False

    def test_default_histogram_enabled(self):
        cfg = CalibrationConfig()
        assert cfg.use_histogram is True

    def test_default_auto_save(self):
        cfg = CalibrationConfig()
        assert cfg.auto_save is True

    def test_roi_fraction_validation(self):
        with pytest.raises(ValidationError):
            CalibrationConfig(roi_fraction=0.01)  # below ge=0.05

    def test_percentile_low_validation(self):
        with pytest.raises(ValidationError):
            CalibrationConfig(percentile_low=30.0)  # above le=25.0

    def test_min_pixels_validation(self):
        with pytest.raises(ValidationError):
            CalibrationConfig(min_pixels=5)  # below ge=10


# -- step() tests --------------------------------------------------------------


class TestStep:
    """Test the step() method."""

    def test_step_inactive_returns_inactive(self):
        cal_cfg = _default_cal_config()
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_bgr()
        state, result = cal.step(frame)
        assert state == CalibrationState.INACTIVE
        assert result is None

    def test_step_collecting_returns_collecting(self):
        cal_cfg = _default_cal_config()
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_bgr()
        cal.start(frame.shape)
        state, result = cal.step(frame)
        assert state == CalibrationState.COLLECTING
        assert result is None

    def test_step_preview_returns_result(self):
        cal_cfg = _default_cal_config(roi_fraction=0.8)
        det_cfg = _default_detection_config()
        cal = AutoCalibrator(cal_cfg, det_cfg)
        frame = _make_blue_frame(100, 100)
        cal.start(frame.shape)
        cal.collect(frame)
        state, result = cal.step(frame)
        assert state == CalibrationState.PREVIEW
        assert result is not None
