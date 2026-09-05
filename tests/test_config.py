"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cloak.config.loader import load_config
from cloak.config.schemas import (
    AdaptiveConfig,
    BackgroundConfig,
    CameraConfig,
    CloakConfig,
    DetectionConfig,
    MaskConfig,
    PerformanceConfig,
    ProcessingConfig,
    RenderingConfig,
    TemporalConfig,
)

# -- schema defaults ----------------------------------------------------------


class TestCameraConfigDefaults:
    def test_default_device_id(self) -> None:
        assert CameraConfig().device_id == 0

    def test_default_resolution(self) -> None:
        cfg = CameraConfig()
        assert cfg.width == 640
        assert cfg.height == 480

    def test_default_fps(self) -> None:
        assert CameraConfig().fps == 30


class TestBackgroundConfigDefaults:
    def test_default_capture_frames(self) -> None:
        assert BackgroundConfig().capture_frames == 30

    def test_default_countdown_seconds(self) -> None:
        assert BackgroundConfig().countdown_seconds == 3.0

    def test_default_aggregation_method(self) -> None:
        assert BackgroundConfig().aggregation_method == "median"


class TestDetectionConfigDefaults:
    def test_default_mode(self) -> None:
        assert DetectionConfig().mode == "hsv"

    def test_hsv_lower_length(self) -> None:
        assert len(DetectionConfig().hsv_lower) == 3

    def test_hsv_upper_length(self) -> None:
        assert len(DetectionConfig().hsv_upper) == 3


class TestProcessingConfigDefaults:
    def test_blur_kernel_is_odd(self) -> None:
        assert ProcessingConfig().blur_kernel % 2 == 1

    def test_morphology_kernel_is_odd(self) -> None:
        assert ProcessingConfig().morphology_kernel % 2 == 1


class TestCloakConfigDefaults:
    def test_all_sections_present(self) -> None:
        cfg = CloakConfig()
        assert isinstance(cfg.camera, CameraConfig)
        assert isinstance(cfg.background, BackgroundConfig)
        assert isinstance(cfg.detection, DetectionConfig)
        assert isinstance(cfg.processing, ProcessingConfig)
        assert isinstance(cfg.mask, MaskConfig)
        assert isinstance(cfg.rendering, RenderingConfig)
        assert isinstance(cfg.temporal, TemporalConfig)
        assert isinstance(cfg.adaptive, AdaptiveConfig)
        assert isinstance(cfg.performance, PerformanceConfig)


# -- schema validation --------------------------------------------------------


class TestSchemaValidation:
    def test_width_too_small_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CameraConfig(width=100)

    def test_fps_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CameraConfig(fps=0)

    def test_hsv_lower_wrong_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DetectionConfig(hsv_lower=[10, 20])

    def test_custom_values_accepted(self) -> None:
        cfg = CameraConfig(device_id=2, width=1920, height=1080, fps=60)
        assert cfg.device_id == 2
        assert cfg.width == 1920


# -- YAML loader --------------------------------------------------------------


class TestLoadConfig:
    def test_load_default_config(self, tmp_path: Path) -> None:
        config_data = {
            "camera": {"device_id": 1, "width": 320, "height": 240, "fps": 15},
            "detection": {"mode": "hsv", "hsv_lower": [0, 0, 0], "hsv_upper": [180, 255, 255]},
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        cfg = load_config(config_file)
        assert cfg.camera.device_id == 1
        assert cfg.camera.width == 320
        assert cfg.detection.hsv_upper == [180, 255, 255]

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, CloakConfig)
        assert cfg.camera.device_id == 0

    def test_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")
        cfg = load_config(config_file)
        assert isinstance(cfg, CloakConfig)

    def test_partial_config_fills_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "partial.yaml"
        config_file.write_text("camera:\n  device_id: 5\n", encoding="utf-8")

        cfg = load_config(config_file)
        assert cfg.camera.device_id == 5
        assert cfg.camera.width == 640  # default
        assert cfg.detection.mode == "hsv"  # default
        assert cfg.background.countdown_seconds == 3.0  # default
        assert cfg.background.aggregation_method == "median"  # default

    def test_background_config_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bg.yaml"
        config_file.write_text(
            "background:\n  capture_frames: 10\n  countdown_seconds: 5.0\n  "
            "aggregation_method: mean\n",
            encoding="utf-8",
        )
        cfg = load_config(config_file)
        assert cfg.background.capture_frames == 10
        assert cfg.background.countdown_seconds == 5.0
        assert cfg.background.aggregation_method == "mean"
