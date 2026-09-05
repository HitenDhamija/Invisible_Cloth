"""Tests for AI model manager using mocked ONNX Runtime."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from cloak.config.schemas import AIConfig
from cloak.detection.model_manager import ModelManager, ModelManagerError

# -- helpers -------------------------------------------------------------------


def _make_bgr(h: int = 240, w: int = 320, bgr: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _make_blue_frame(h: int = 240, w: int = 320) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[60:180, 80:240] = (255, 0, 0)  # blue in BGR
    return frame


def _default_ai_config(**kwargs) -> AIConfig:
    defaults = dict(
        enabled=True,
        inference_width=320,
        inference_height=240,
        confidence_threshold=0.5,
    )
    defaults.update(kwargs)
    return AIConfig(**defaults)


# -- ModelManager tests --------------------------------------------------------


class TestModelManagerNotLoaded:
    """Model manager state before loading."""

    def test_available_is_false(self):
        cfg = _default_ai_config()
        mgr = ModelManager(cfg)
        assert mgr.available is False

    def test_device_info_not_loaded(self):
        cfg = _default_ai_config()
        mgr = ModelManager(cfg)
        assert mgr.device_info == "not loaded"

    def test_last_latency_zero(self):
        cfg = _default_ai_config()
        mgr = ModelManager(cfg)
        assert mgr.last_latency_ms == 0.0

    def test_frame_count_zero(self):
        cfg = _default_ai_config()
        mgr = ModelManager(cfg)
        assert mgr.frame_count == 0


class TestModelManagerNoOnnxRuntime:
    """Returns error when onnxruntime is not installed."""

    def test_ensure_loaded_raises(self):
        cfg = _default_ai_config()
        mgr = ModelManager(cfg)
        with (
            patch.dict("sys.modules", {"onnxruntime": None}),
            pytest.raises(ModelManagerError, match="onnxruntime"),
        ):
            mgr.ensure_loaded()


class TestModelManagerClose:
    """Close releases resources without error."""

    def test_close_on_uninitialized(self):
        cfg = _default_ai_config()
        mgr = ModelManager(cfg)
        mgr.close()  # should not raise

    def test_double_close(self):
        cfg = _default_ai_config()
        mgr = ModelManager(cfg)
        mgr.close()
        mgr.close()  # should not raise


class TestModelManagerPostprocess:
    """Test postprocessing of model outputs."""

    def test_postprocess_4d_output(self):
        """4D output (1, C, H, W) should extract class 0."""
        cfg = _default_ai_config(inference_width=160, inference_height=120)
        mgr = ModelManager(cfg)
        mgr._loaded = True  # skip actual loading

        # Create fake 4D output with person class activated
        output = np.zeros((1, 2, 120, 160), dtype=np.float32)
        output[0, 0, 30:90, 40:120] = 0.9  # person class confidence

        mask = mgr._postprocess([output], 240, 320)
        assert mask.shape == (240, 320)
        assert mask.dtype == np.uint8
        assert np.any(mask == 255)

    def test_postprocess_2d_output(self):
        """2D output (H, W) should threshold directly."""
        cfg = _default_ai_config(confidence_threshold=0.3)
        mgr = ModelManager(cfg)
        mgr._loaded = True

        output = np.zeros((80, 100), dtype=np.float32)
        output[20:60, 30:70] = 0.8

        mask = mgr._postprocess([output], 240, 320)
        assert mask.shape == (240, 320)
        assert np.any(mask == 255)

    def test_postprocess_empty_detections(self):
        """YOLO output with no detections returns empty mask."""
        cfg = _default_ai_config()
        mgr = ModelManager(cfg)
        mgr._loaded = True

        # YOLO format: (1, 4+nc, num_anchors) with person score below threshold
        output = np.zeros((1, 6, 100), dtype=np.float32)
        output[0, 4, :] = 0.1  # person scores all low

        mask = mgr._postprocess([output], 240, 320)
        assert mask.shape == (240, 320)
        assert np.sum(mask) == 0

    def test_postprocess_clips_to_01(self):
        """Values outside [0,1] should be clipped."""
        cfg = _default_ai_config(confidence_threshold=0.5)
        mgr = ModelManager(cfg)
        mgr._loaded = True

        output = np.full((80, 100), 1.5, dtype=np.float32)
        mask = mgr._postprocess([output], 240, 320)
        assert np.all((mask == 0) | (mask == 255))


class TestModelManagerConfigDefaults:
    """AIConfig has correct defaults for hybrid settings."""

    def test_default_hybrid_model_path(self):
        cfg = AIConfig()
        assert cfg.hybrid_model_path == ""

    def test_default_inference_width(self):
        cfg = AIConfig()
        assert cfg.inference_width == 320

    def test_default_inference_height(self):
        cfg = AIConfig()
        assert cfg.inference_height == 240

    def test_default_frame_skip(self):
        cfg = AIConfig()
        assert cfg.inference_frame_skip == 1

    def test_default_half_precision(self):
        cfg = AIConfig()
        assert cfg.use_half_precision is False

    def test_default_confidence_threshold(self):
        cfg = AIConfig()
        assert cfg.confidence_threshold == 0.5

    def test_inference_width_validation(self):
        with pytest.raises(ValidationError):
            AIConfig(inference_width=80)  # below ge=160

    def test_inference_height_validation(self):
        with pytest.raises(ValidationError):
            AIConfig(inference_height=60)  # below ge=120

    def test_frame_skip_validation(self):
        with pytest.raises(ValidationError):
            AIConfig(inference_frame_skip=0)  # below ge=1

    def test_confidence_threshold_validation(self):
        with pytest.raises(ValidationError):
            AIConfig(confidence_threshold=0.0)  # below ge=0.1
