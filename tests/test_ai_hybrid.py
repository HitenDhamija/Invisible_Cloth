"""Tests for AI hybrid detection using mocked model manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from cloak.config.schemas import AIConfig, DetectionConfig, ProcessingConfig
from cloak.detection.segmenter import AIHybridDetector

# -- helpers -------------------------------------------------------------------


def _make_bgr(h: int = 100, w: int = 100, bgr: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _make_blue_frame(h: int = 100, w: int = 100) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[20:80, 20:80] = (255, 0, 0)  # blue in BGR
    return frame


def _default_ai_config(**kwargs) -> AIConfig:
    defaults = dict(enabled=True, fallback_to_hsv=True)
    defaults.update(kwargs)
    return AIConfig(**defaults)


def _default_detection_config(**kwargs) -> DetectionConfig:
    return DetectionConfig(**kwargs)


def _default_processing_config(**kwargs) -> ProcessingConfig:
    return ProcessingConfig(**kwargs)


# -- AIHybridDetector tests ---------------------------------------------------


class TestAIHybridFallback:
    """Falls back to pure HSV when AI model unavailable."""

    def test_fallback_returns_hsv_mask(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.predict.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_blue_frame()
            mask, stats = hybrid.detect(frame)

            # Should get HSV mask (blue region detected)
            assert stats.cloak_pixels > 0

    def test_no_fallback_when_disabled(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=False)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.predict.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_blue_frame()
            mask, stats = hybrid.detect(frame)

            # No person, no fallback -> empty mask
            assert stats.cloak_pixels == 0


class TestAIHybridIntersection:
    """Intersection constrains blue mask to AI person region."""

    def test_intersection_with_person_mask(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            # Person mask covers center region
            person_mask = np.zeros((100, 100), dtype=np.uint8)
            person_mask[30:70, 30:70] = 255
            mock_mm.predict.return_value = person_mask
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_blue_frame()  # blue in 20:80, 20:80
            mask, stats = hybrid.detect(frame)

            # Blue (20:80, 20:80) AND person (30:70, 30:70) = 30:70, 30:70
            assert stats.cloak_pixels > 0
            assert stats.cloak_pixels < (60 * 60)

    def test_intersection_with_no_overlap(self):
        """Blue in top-left, person in bottom-right -> no overlap."""
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            person_mask = np.zeros((100, 100), dtype=np.uint8)
            person_mask[80:100, 80:100] = 255  # bottom-right
            mock_mm.predict.return_value = person_mask
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_blue_frame()  # blue in 20:80, 20:80
            mask, stats = hybrid.detect(frame)

            # No overlap -> zero cloak pixels
            assert stats.cloak_pixels == 0


class TestAIHybridInterface:
    """Preserves same interface as BlueColorDetector."""

    def test_returns_tuple(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.predict.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_bgr()
            result = hybrid.detect(frame)

            assert isinstance(result, tuple)
            assert len(result) == 2
            mask, stats = result
            assert mask.dtype == np.uint8
            assert hasattr(stats, "cloak_ratio")

    def test_set_bounds_delegates(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.predict.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            hybrid.set_bounds([80, 100, 100], [140, 255, 255])

            np.testing.assert_array_equal(hybrid.lower_bound, [80, 100, 100])
            np.testing.assert_array_equal(hybrid.upper_bound, [140, 255, 255])

    def test_close_releases(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            hybrid.close()
            mock_mm.close.assert_called_once()


class TestAIHybridFrameSkipping:
    """Frame skipping caches AI results between inference frames."""

    def test_skips_frames(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True, inference_frame_skip=3)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            person_mask = np.zeros((100, 100), dtype=np.uint8)
            person_mask[30:70, 30:70] = 255
            mock_mm.predict.return_value = person_mask
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_blue_frame()

            # Frame 1: counter=1, 1%3=1!=0 but cache is None → inference (call 1)
            hybrid.detect(frame)
            assert mock_mm.predict.call_count == 1

            # Frame 2: counter=2, 2%3=2!=0, cache exists → no inference
            hybrid.detect(frame)
            assert mock_mm.predict.call_count == 1

            # Frame 3: counter=3, 3%3=0 → inference (call 2)
            hybrid.detect(frame)
            assert mock_mm.predict.call_count == 2

            # Frame 4: counter=4, 4%3=1!=0, cache exists → no inference
            hybrid.detect(frame)
            assert mock_mm.predict.call_count == 2

            # Frame 5: counter=5, 5%3=2!=0, cache exists → no inference
            hybrid.detect(frame)
            assert mock_mm.predict.call_count == 2

            # Frame 6: counter=6, 6%3=0 → inference (call 3)
            hybrid.detect(frame)
            assert mock_mm.predict.call_count == 3


class TestAIHybridLatencyTracking:
    """Tracks AI inference latency for benchmarking."""

    def test_latency_reported(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.predict.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_mm.last_latency_ms = 5.5
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_bgr()
            hybrid.detect(frame)

            assert hybrid.last_ai_latency_ms == 5.5


class TestAIHybridPersonMask:
    """Exposes last person mask for debugging."""

    def test_person_mask_available_after_detect(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            person_mask = np.zeros((100, 100), dtype=np.uint8)
            person_mask[30:70, 30:70] = 255
            mock_mm.predict.return_value = person_mask
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            frame = _make_blue_frame()
            hybrid.detect(frame)

            assert hybrid.last_person_mask is not None
            assert hybrid.last_person_mask.shape == (100, 100)

    def test_person_mask_none_before_detect(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True, fallback_to_hsv=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            assert hybrid.last_person_mask is None


class TestAIHybridModelManagerExposure:
    """Exposes model manager for external inspection."""

    def test_model_manager_property(self):
        det_cfg = _default_detection_config()
        proc_cfg = _default_processing_config()
        ai_cfg = _default_ai_config(enabled=True)

        with patch("cloak.detection.segmenter.ModelManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm_class.return_value = mock_mm

            hybrid = AIHybridDetector(det_cfg, proc_cfg, ai_cfg)
            assert hybrid.model_manager is mock_mm
