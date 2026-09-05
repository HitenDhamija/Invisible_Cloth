"""Offline integration tests for the full invisibility cloak pipeline.

Tests run without a webcam by synthesizing frames with blue rectangles.
Pipeline: synthetic image -> HSV segmentation -> mask refinement -> rendering.
"""

from __future__ import annotations

import cv2
import numpy as np

from cloak.config.schemas import (
    AdaptiveConfig,
    DetectionConfig,
    MaskConfig,
    ProcessingConfig,
    RenderingConfig,
    TemporalConfig,
)
from cloak.detection.adaptive import AdaptivePreprocessor
from cloak.detection.detector import BlueColorDetector
from cloak.processing.refiner import MaskRefiner
from cloak.processing.temporal import TemporalMaskSmoother
from cloak.rendering.renderer import InvisibilityRenderer

FRAME_W, FRAME_H = 640, 480
BLUE_BGR = (255, 50, 30)  # BGR for a distinct blue
BG_GRAY = 128


def _make_frame(
    blue_rect: tuple[int, int, int, int] | None = None,
    bg_value: int = BG_GRAY,
) -> np.ndarray:
    """Create a BGR frame with an optional blue rectangle."""
    frame = np.full((FRAME_H, FRAME_W, 3), bg_value, dtype=np.uint8)
    if blue_rect is not None:
        x, y, w, h = blue_rect
        frame[y : y + h, x : x + w] = BLUE_BGR
    return frame


def _make_background(bg_value: int = BG_GRAY) -> np.ndarray:
    return np.full((FRAME_H, FRAME_W, 3), bg_value, dtype=np.uint8)


def _pipeline_configs(
    soft_blend: bool = False,
    feather_radius: int = 7,
) -> tuple[DetectionConfig, ProcessingConfig, MaskConfig, RenderingConfig, TemporalConfig]:
    return (
        DetectionConfig(hsv_lower=[85, 100, 100], hsv_upper=[135, 255, 255]),
        ProcessingConfig(blur_kernel=5, morphology_kernel=5),
        MaskConfig(
            median_kernel=5,
            morphology_kernel=5,
            min_region_area=100,
            feather_radius=feather_radius,
        ),
        RenderingConfig(use_soft_blend=soft_blend),
        TemporalConfig(enabled=True, ema_alpha=0.6, persistence_frames=3),
    )


class TestBasicPipeline:
    """Full pipeline with a synthetic frame containing a blue rectangle."""

    def test_blue_region_replaced(self) -> None:
        det_cfg, proc_cfg, mask_cfg, render_cfg, _ = _pipeline_configs()
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)

        blue_rect = (200, 150, 120, 180)
        frame = _make_frame(blue_rect=blue_rect)
        background = _make_background()

        mask, _ = detector.detect(frame)
        binary, soft, _ = refiner.refine(mask)
        output = renderer.render(frame, background, binary, soft)

        # Check interior of the blue rectangle was replaced with background.
        # Shrink by a few pixels to account for morphological erosion at edges.
        x, y, w, h = blue_rect
        margin = 5
        region = output[y + margin : y + h - margin, x + margin : x + w - margin]
        bg_region = background[y + margin : y + h - margin, x + margin : x + w - margin]
        np.testing.assert_array_equal(region, bg_region)

    def test_output_shape_matches_input(self) -> None:
        det_cfg, proc_cfg, mask_cfg, render_cfg, _ = _pipeline_configs()
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)

        frame = _make_frame(blue_rect=(200, 150, 120, 180))
        background = _make_background()

        mask, _ = detector.detect(frame)
        binary, soft, _ = refiner.refine(mask)
        output = renderer.render(frame, background, binary, soft)

        assert output.shape == frame.shape

    def test_output_dtype_is_uint8(self) -> None:
        det_cfg, proc_cfg, mask_cfg, render_cfg, _ = _pipeline_configs()
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)

        frame = _make_frame(blue_rect=(200, 150, 120, 180))
        background = _make_background()

        mask, _ = detector.detect(frame)
        binary, soft, _ = refiner.refine(mask)
        output = renderer.render(frame, background, binary, soft)

        assert output.dtype == np.uint8


class TestPipelineWithTemporalSmoothing:
    """Pipeline with temporal smoother across multiple frames."""

    def test_masks_stabilize_over_frames(self) -> None:
        det_cfg, proc_cfg, mask_cfg, render_cfg, temporal_cfg = _pipeline_configs()
        detector = BlueColorDetector(det_cfg, proc_cfg)
        smoother = TemporalMaskSmoother(temporal_cfg)

        blue_rect = (200, 150, 120, 180)

        pixel_variances: list[float] = []
        prev_mask: np.ndarray | None = None

        for _ in range(10):
            frame = _make_frame(blue_rect=blue_rect)
            raw_mask, _ = detector.detect(frame)
            smooth_mask = smoother.smooth(raw_mask)

            if prev_mask is not None:
                diff = np.abs(smooth_mask.astype(np.float32) - prev_mask.astype(np.float32))
                pixel_variances.append(float(np.mean(diff)))

            prev_mask = smooth_mask

        # Later frames should have lower pixel-level variation
        assert len(pixel_variances) >= 5
        first_half = np.mean(pixel_variances[: len(pixel_variances) // 2])
        second_half = np.mean(pixel_variances[len(pixel_variances) // 2 :])
        assert second_half <= first_half

    def test_output_is_valid(self) -> None:
        det_cfg, proc_cfg, mask_cfg, render_cfg, temporal_cfg = _pipeline_configs()
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)
        smoother = TemporalMaskSmoother(temporal_cfg)

        background = _make_background()
        blue_rect = (200, 150, 120, 180)

        for _ in range(10):
            frame = _make_frame(blue_rect=blue_rect)
            raw_mask, _ = detector.detect(frame)
            smooth_mask = smoother.smooth(raw_mask)
            binary, soft, _ = refiner.refine(smooth_mask)
            output = renderer.render(frame, background, binary, soft)

            assert output.shape == frame.shape
            assert output.dtype == np.uint8


class TestPipelineWithSoftBlend:
    """Pipeline using soft blend rendering."""

    def test_soft_blend_output_valid(self) -> None:
        det_cfg, proc_cfg, mask_cfg, render_cfg, _ = _pipeline_configs(
            soft_blend=True,
            feather_radius=7,
        )
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)

        frame = _make_frame(blue_rect=(200, 150, 120, 180))
        background = _make_background()

        mask, _ = detector.detect(frame)
        binary, soft, _ = refiner.refine(mask)
        output = renderer.render(frame, background, binary, soft)

        assert output.shape == frame.shape
        assert output.dtype == np.uint8

    def test_soft_blend_edges_smoother(self) -> None:
        det_cfg, proc_cfg, mask_cfg, _, _ = _pipeline_configs()
        hard_render_cfg = RenderingConfig(use_soft_blend=False)
        soft_render_cfg = RenderingConfig(use_soft_blend=True)
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        hard_renderer = InvisibilityRenderer(hard_render_cfg)
        soft_renderer = InvisibilityRenderer(soft_render_cfg)

        frame = _make_frame(blue_rect=(200, 150, 120, 180))
        background = _make_background()

        mask, _ = detector.detect(frame)
        binary, soft, _ = refiner.refine(mask)
        hard_out = hard_renderer.render(frame, background, binary, soft)
        soft_out = soft_renderer.render(frame, background, binary, soft)

        # Both outputs should be valid
        assert hard_out.shape == frame.shape
        assert soft_out.shape == frame.shape
        assert hard_out.dtype == np.uint8
        assert soft_out.dtype == np.uint8

        # Outputs should differ (soft blend produces different pixel values)
        assert not np.array_equal(hard_out, soft_out)

        # Soft blend should have intermediate values at edges (not just 0 or 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(binary, kernel)
        boundary = (dilated > 0) & (binary == 0)
        if np.any(boundary):
            soft_boundary = soft_out[boundary]
            has_intermediate = np.any((soft_boundary > 10) & (soft_boundary < 245))
            assert has_intermediate, "Soft blend should produce intermediate values at edges"


class TestPipelineWithAdaptivePreprocessing:
    """Pipeline with adaptive preprocessing for different lighting."""

    def test_dark_frame_detected(self) -> None:
        adaptive_cfg = AdaptiveConfig(enabled=True, clahe_clip=2.0, clahe_grid=8)
        det_cfg, proc_cfg, mask_cfg, render_cfg, _ = _pipeline_configs()
        preprocessor = AdaptivePreprocessor(adaptive_cfg)
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)

        # Dark frame: low background value
        dark_frame = _make_frame(blue_rect=(200, 150, 120, 180), bg_value=30)
        dark_processed = preprocessor.preprocess(dark_frame)
        mask, stats = detector.detect(dark_processed)
        binary, soft, _ = refiner.refine(mask)

        assert stats.cloak_pixels > 0, "Blue rectangle not detected in dark frame"

        background = _make_background(bg_value=30)
        output = renderer.render(dark_frame, background, binary, soft)
        assert output.shape == dark_frame.shape
        assert output.dtype == np.uint8

    def test_bright_frame_detected(self) -> None:
        adaptive_cfg = AdaptiveConfig(enabled=True, clahe_clip=2.0, clahe_grid=8)
        det_cfg, proc_cfg, mask_cfg, render_cfg, _ = _pipeline_configs()
        preprocessor = AdaptivePreprocessor(adaptive_cfg)
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)

        # Bright frame: high background value
        bright_frame = _make_frame(blue_rect=(200, 150, 120, 180), bg_value=220)
        bright_processed = preprocessor.preprocess(bright_frame)
        mask, stats = detector.detect(bright_processed)
        binary, soft, _ = refiner.refine(mask)

        assert stats.cloak_pixels > 0, "Blue rectangle not detected in bright frame"

        background = _make_background(bg_value=220)
        output = renderer.render(bright_frame, background, binary, soft)
        assert output.shape == bright_frame.shape
        assert output.dtype == np.uint8

    def test_both_lighting_conditions_work(self) -> None:
        adaptive_cfg = AdaptiveConfig(enabled=True)
        det_cfg, proc_cfg, mask_cfg, render_cfg, _ = _pipeline_configs()
        preprocessor = AdaptivePreprocessor(adaptive_cfg)
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)

        results: list[int] = []
        for bg_val in [30, 128, 220]:
            frame = _make_frame(blue_rect=(200, 150, 120, 180), bg_value=bg_val)
            processed = preprocessor.preprocess(frame)
            mask, stats = detector.detect(processed)
            binary, soft, _ = refiner.refine(mask)
            background = _make_background(bg_value=bg_val)
            output = renderer.render(frame, background, binary, soft)
            assert output.shape == frame.shape
            results.append(stats.cloak_pixels)

        # All three lighting conditions should detect blue pixels
        assert all(r > 0 for r in results)


class TestEndToEndMultipleFrames:
    """Simulate 30 frames of video with a moving blue rectangle."""

    def test_30_frames_no_crash(self) -> None:
        det_cfg, proc_cfg, mask_cfg, render_cfg, temporal_cfg = _pipeline_configs()
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)
        smoother = TemporalMaskSmoother(temporal_cfg)

        background = _make_background()
        num_frames = 30

        for i in range(num_frames):
            # Move the blue rectangle slightly each frame
            x_offset = 200 + i * 2
            y_offset = 150 + i
            frame = _make_frame(blue_rect=(x_offset, y_offset, 120, 180))

            raw_mask, stats = detector.detect(frame)
            smooth_mask = smoother.smooth(raw_mask)
            binary, soft, _ = refiner.refine(smooth_mask)
            output = renderer.render(frame, background, binary, soft)

            assert output.shape == frame.shape, f"Frame {i}: shape mismatch"
            assert output.dtype == np.uint8, f"Frame {i}: dtype mismatch"

    def test_all_outputs_valid(self) -> None:
        det_cfg, proc_cfg, mask_cfg, render_cfg, temporal_cfg = _pipeline_configs()
        detector = BlueColorDetector(det_cfg, proc_cfg)
        refiner = MaskRefiner(mask_cfg)
        renderer = InvisibilityRenderer(render_cfg)
        smoother = TemporalMaskSmoother(temporal_cfg)

        background = _make_background()
        outputs: list[np.ndarray] = []

        for i in range(30):
            x_offset = 200 + (i % 20) * 3
            y_offset = 150 + (i % 15) * 2
            frame = _make_frame(blue_rect=(x_offset, y_offset, 120, 180))

            raw_mask, _ = detector.detect(frame)
            smooth_mask = smoother.smooth(raw_mask)
            binary, soft, _ = refiner.refine(smooth_mask)
            output = renderer.render(frame, background, binary, soft)
            outputs.append(output)

        assert len(outputs) == 30
        for out in outputs:
            assert out is not None
            assert out.shape == (FRAME_H, FRAME_W, 3)
            assert out.dtype == np.uint8
