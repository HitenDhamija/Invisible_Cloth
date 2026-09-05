"""Tests for person-aware detection using mocked MediaPipe."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from cloak.config.schemas import AIConfig, DetectionConfig, ProcessingConfig
from cloak.detection.person import PersonDetector
from cloak.detection.person_aware import PersonAwareDetector

# -- helpers -------------------------------------------------------------------


def _make_bgr(
    h: int = 100, w: int = 100, bgr: tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _make_blue_frame(h: int = 100, w: int = 100) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[20:80, 20:80] = (255, 0, 0)  # blue in BGR
    return frame


def _default_ai_config(**kwargs) -> AIConfig:
    return AIConfig(**kwargs)


def _default_detection_config(**kwargs) -> DetectionConfig:
    return DetectionConfig(**kwargs)


def _default_processing_config(**kwargs) -> ProcessingConfig:
    return ProcessingConfig(**kwargs)


# -- PersonDetector tests ------------------------------------------------------


class TestPersonDetectorDisabled:
    """Returns zeros when disabled."""

    def test_returns_zeros(self):
        cfg = _default_ai_config(enabled=False)
        detector = PersonDetector(cfg)
        frame = _make_bgr()
        mask = detector.detect(frame, timestamp_ms=0)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32
        assert np.sum(mask) == 0.0

    def test_available_is_false(self):
        cfg = _default_ai_config(enabled=False)
        detector = PersonDetector(cfg)
        assert detector.available is False


class TestPersonDetectorNoMediaPipe:
    """Returns zeros when mediapipe is not installed."""

    def test_returns_zeros_on_import_error(self):
        cfg = _default_ai_config(enabled=True)
        detector = PersonDetector(cfg)
        frame = _make_bgr()
        mask = detector.detect(frame, timestamp_ms=0)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32
        assert np.sum(mask) == 0.0


class TestPersonDetectorClose:
    """Close releases resources without error."""

    def test_close_on_uninitialized(self):
        cfg = _default_ai_config(enabled=False)
        detector = PersonDetector(cfg)
        detector.close()  # should not raise

    def test_double_close(self):
        cfg = _default_ai_config(enabled=False)
        detector = PersonDetector(cfg)
        detector.close()
        detector.close()  # should not raise


# -- PersonAwareDetector tests -------------------------------------------------


class TestPersonAwareFallback:
    """Falls back to pure HSV when no person detected."""

    def test_fallback_returns_hsv_mask(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.person_aware.PersonDetector") as mock_pd_class:
            mock_pd = MagicMock()
            # Person detector returns all zeros (no person)
            mock_pd.detect.return_value = np.zeros((100, 100), dtype=np.float32)
            mock_pd_class.return_value = mock_pd

            pad = PersonAwareDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_blue_frame()
            mask, stats = pad.detect(frame)

            # Should get the HSV mask (blue region detected)
            assert stats.cloak_pixels > 0

    def test_no_fallback_when_disabled(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=False)

        with patch("cloak.detection.person_aware.PersonDetector") as mock_pd_class:
            mock_pd = MagicMock()
            mock_pd.detect.return_value = np.zeros((100, 100), dtype=np.float32)
            mock_pd_class.return_value = mock_pd

            pad = PersonAwareDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_blue_frame()
            mask, stats = pad.detect(frame)

            # No person, no fallback -> mask should be all zeros
            assert stats.cloak_pixels == 0


class TestPersonAwareIntersection:
    """Intersection constrains blue mask to person region."""

    def test_intersection_with_person_mask(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.person_aware.PersonDetector") as mock_pd_class:
            mock_pd = MagicMock()
            # Person mask covers center region
            person_mask = np.zeros((100, 100), dtype=np.float32)
            person_mask[30:70, 30:70] = 0.9  # person in center
            mock_pd.detect.return_value = person_mask
            mock_pd_class.return_value = mock_pd

            pad = PersonAwareDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_blue_frame()  # blue in 20:80, 20:80
            mask, stats = pad.detect(frame)

            # Blue region (20:80, 20:80) AND person (30:70, 30:70) = 30:70, 30:70
            assert stats.cloak_pixels > 0
            # The constrained region should be smaller than the full blue region
            assert stats.cloak_pixels < (60 * 60)  # less than full blue rect


class TestPersonAwareInterface:
    """Preserves same interface as BlueColorDetector."""

    def test_returns_tuple(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.person_aware.PersonDetector") as mock_pd_class:
            mock_pd = MagicMock()
            mock_pd.detect.return_value = np.zeros((100, 100), dtype=np.float32)
            mock_pd_class.return_value = mock_pd

            pad = PersonAwareDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_bgr()
            result = pad.detect(frame)

            assert isinstance(result, tuple)
            assert len(result) == 2
            mask, stats = result
            assert mask.dtype == np.uint8
            assert hasattr(stats, "cloak_ratio")

    def test_set_bounds_delegates(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True)

        with patch("cloak.detection.person_aware.PersonDetector") as mock_pd_class:
            mock_pd = MagicMock()
            mock_pd.detect.return_value = np.zeros((100, 100), dtype=np.float32)
            mock_pd_class.return_value = mock_pd

            pad = PersonAwareDetector(det_cfg, proc_cfg, ai_cfg)
            pad.set_bounds([80, 100, 100], [140, 255, 255])

            # HSV detector bounds should be updated
            np.testing.assert_array_equal(pad.lower_bound, [80, 100, 100])
            np.testing.assert_array_equal(pad.upper_bound, [140, 255, 255])

    def test_close_releases(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True)

        with patch("cloak.detection.person_aware.PersonDetector") as mock_pd_class:
            mock_pd = MagicMock()
            mock_pd_class.return_value = mock_pd

            pad = PersonAwareDetector(det_cfg, proc_cfg, ai_cfg)
            pad.close()
            mock_pd.close.assert_called_once()


# -- Config schema tests -------------------------------------------------------


class TestAIConfigDefaults:
    """AIConfig has correct defaults."""

    def test_default_enabled(self):
        cfg = AIConfig()
        assert cfg.enabled is False

    def test_default_fallback(self):
        cfg = AIConfig()
        assert cfg.fallback_to_hsv is True

    def test_default_threshold(self):
        cfg = AIConfig()
        assert cfg.person_threshold == 0.5

    def test_default_model_complexity(self):
        cfg = AIConfig()
        assert cfg.model_complexity == 0

    def test_threshold_validation(self):
        with pytest.raises(ValidationError):
            AIConfig(person_threshold=0.0)  # below ge=0.1

    def test_model_complexity_validation(self):
        with pytest.raises(ValidationError):
            AIConfig(model_complexity=3)  # above le=2


class TestCloakConfigAI:
    """CloakConfig includes AI section."""

    def test_ai_section_present(self):
        from cloak.config.schemas import CloakConfig

        cfg = CloakConfig()
        assert hasattr(cfg, "ai")
        assert isinstance(cfg.ai, AIConfig)
