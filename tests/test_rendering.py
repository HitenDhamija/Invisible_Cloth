"""Tests for the invisibility renderer using synthetic images (no webcam)."""

from __future__ import annotations

import numpy as np
import pytest

from cloak.config.schemas import RenderingConfig
from cloak.rendering.renderer import InvisibilityRenderer, RenderError

# -- helpers -------------------------------------------------------------------


def _solid_bgr(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _mask_rect(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _soft_from_binary(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    import cv2
    k = radius * 2 + 1
    soft = mask.astype(np.float32) / 255.0
    return cv2.GaussianBlur(soft, (k, k), 0)


# -- fixtures ------------------------------------------------------------------


@pytest.fixture
def hard_renderer() -> InvisibilityRenderer:
    return InvisibilityRenderer(RenderingConfig(use_soft_blend=False))


@pytest.fixture
def soft_renderer() -> InvisibilityRenderer:
    return InvisibilityRenderer(RenderingConfig(use_soft_blend=True, blend_alpha=0.5))


# -- hard composite tests -----------------------------------------------------


class TestHardComposite:
    def test_masked_region_from_background(self, hard_renderer: InvisibilityRenderer) -> None:
        """Pixels where mask=255 should come from the background."""
        bg = _solid_bgr(80, 80, (200, 0, 0))    # blue background
        frame = _solid_bgr(80, 80, (0, 0, 200))  # red live frame
        mask = _mask_rect(80, 80, 20, 60, 20, 60)

        output = hard_renderer.render(frame, bg, mask)

        # Inside the mask rectangle: should be background (blue)
        assert np.all(output[30, 30] == [200, 0, 0])
        # Outside the mask: should be live frame (red)
        assert np.all(output[5, 5] == [0, 0, 200])

    def test_unmasked_region_from_live_frame(self, hard_renderer: InvisibilityRenderer) -> None:
        """Pixels where mask=0 should remain from the live frame."""
        bg = _solid_bgr(60, 60, (100, 100, 100))
        frame = _solid_bgr(60, 60, (50, 50, 50))
        mask = _mask_rect(60, 60, 0, 30, 0, 30)

        output = hard_renderer.render(frame, bg, mask)

        # Bottom-right (outside mask): live frame
        assert np.all(output[45, 45] == [50, 50, 50])
        # Top-left (inside mask): background
        assert np.all(output[10, 10] == [100, 100, 100])

    def test_empty_mask_returns_original(self, hard_renderer: InvisibilityRenderer) -> None:
        """All-zero mask → output equals the live frame."""
        bg = _solid_bgr(40, 40, (255, 0, 0))
        frame = _solid_bgr(40, 40, (0, 255, 0))
        mask = np.zeros((40, 40), dtype=np.uint8)

        output = hard_renderer.render(frame, bg, mask)
        np.testing.assert_array_equal(output, frame)

    def test_full_mask_returns_background(self, hard_renderer: InvisibilityRenderer) -> None:
        """All-255 mask → output equals the background."""
        bg = _solid_bgr(40, 40, (255, 0, 0))
        frame = _solid_bgr(40, 40, (0, 255, 0))
        mask = np.full((40, 40), 255, dtype=np.uint8)

        output = hard_renderer.render(frame, bg, mask)
        np.testing.assert_array_equal(output, bg)

    def test_output_shape_matches_input(self, hard_renderer: InvisibilityRenderer) -> None:
        bg = _solid_bgr(100, 120, (0, 0, 0))
        frame = _solid_bgr(100, 120, (0, 0, 0))
        mask = np.zeros((100, 120), dtype=np.uint8)
        output = hard_renderer.render(frame, bg, mask)
        assert output.shape == frame.shape

    def test_no_pixel_overflow(self, hard_renderer: InvisibilityRenderer) -> None:
        """Output values must stay in uint8 range (no arithmetic overflow)."""
        bg = _solid_bgr(50, 50, (255, 255, 255))
        frame = _solid_bgr(50, 50, (255, 255, 255))
        mask = _mask_rect(50, 50, 0, 50, 0, 50)
        output = hard_renderer.render(frame, bg, mask)
        assert output.dtype == np.uint8
        assert int(np.max(output)) <= 255

    def test_output_is_uint8(self, hard_renderer: InvisibilityRenderer) -> None:
        bg = _solid_bgr(30, 30, (0, 0, 0))
        frame = _solid_bgr(30, 30, (0, 0, 0))
        mask = np.zeros((30, 30), dtype=np.uint8)
        output = hard_renderer.render(frame, bg, mask)
        assert output.dtype == np.uint8


# -- soft blend tests ---------------------------------------------------------


class TestSoftBlend:
    def test_masked_region_blends_toward_background(self, soft_renderer: InvisibilityRenderer) -> None:
        """Inside the mask, output should be closer to background."""
        bg = _solid_bgr(60, 60, (200, 200, 200))
        frame = _solid_bgr(60, 60, (50, 50, 50))
        mask = _mask_rect(60, 60, 10, 50, 10, 50)

        output = soft_renderer.render(frame, bg, mask)

        # Inside mask: should be between bg and frame, closer to bg
        center_val = int(output[30, 30, 0])
        assert center_val > 100  # closer to 200 than to 50

    def test_unmasked_region_stays_as_frame(self, soft_renderer: InvisibilityRenderer) -> None:
        """Outside the mask, output should equal the live frame."""
        bg = _solid_bgr(60, 60, (200, 200, 200))
        frame = _solid_bgr(60, 60, (50, 50, 50))
        mask = _mask_rect(60, 60, 10, 50, 10, 50)

        output = soft_renderer.render(frame, bg, mask)

        # Outside mask: exactly the live frame
        assert np.all(output[5, 5] == [50, 50, 50])

    def test_soft_with_feathered_mask(self, soft_renderer: InvisibilityRenderer) -> None:
        """Using a feathered soft mask produces smooth gradients."""
        bg = _solid_bgr(80, 80, (200, 200, 200))
        frame = _solid_bgr(80, 80, (50, 50, 50))
        binary = _mask_rect(80, 80, 20, 60, 20, 60)
        soft = _soft_from_binary(binary, radius=7)

        output = soft_renderer.render(frame, bg, binary, soft_mask=soft)

        # Edge pixel should be between bg and frame (intermediate value)
        edge_val = int(output[20, 40, 0])
        assert 50 < edge_val < 200

    def test_soft_fallback_to_binary_when_no_soft_mask(self) -> None:
        """When soft_mask is None, soft blend uses normalized binary mask."""
        renderer = InvisibilityRenderer(RenderingConfig(use_soft_blend=True))
        bg = _solid_bgr(50, 50, (200, 200, 200))
        frame = _solid_bgr(50, 50, (50, 50, 50))
        mask = _mask_rect(50, 50, 10, 40, 10, 40)

        output = renderer.render(frame, bg, mask)

        # Inside mask: 1.0 * 200 = 200
        assert np.all(output[25, 25] == [200, 200, 200])
        # Outside mask: 0.0 * 200 + 1.0 * 50 = 50
        assert np.all(output[5, 5] == [50, 50, 50])


# -- validation tests ---------------------------------------------------------


class TestValidation:
    def test_mismatched_frame_background_raises(self, hard_renderer: InvisibilityRenderer) -> None:
        bg = _solid_bgr(80, 80, (0, 0, 0))
        frame = _solid_bgr(60, 60, (0, 0, 0))
        mask = np.zeros((60, 60), dtype=np.uint8)
        with pytest.raises(RenderError, match="shape"):
            hard_renderer.render(frame, bg, mask)

    def test_mismatched_mask_shape_raises(self, hard_renderer: InvisibilityRenderer) -> None:
        bg = _solid_bgr(60, 60, (0, 0, 0))
        frame = _solid_bgr(60, 60, (0, 0, 0))
        mask = np.zeros((40, 40), dtype=np.uint8)
        with pytest.raises(RenderError, match="shape"):
            hard_renderer.render(frame, bg, mask)

    def test_valid_dimensions_no_error(self, hard_renderer: InvisibilityRenderer) -> None:
        bg = _solid_bgr(100, 100, (0, 0, 0))
        frame = _solid_bgr(100, 100, (0, 0, 0))
        mask = np.zeros((100, 100), dtype=np.uint8)
        output = hard_renderer.render(frame, bg, mask)
        assert output.shape == (100, 100, 3)


# -- toggle tests -------------------------------------------------------------


class TestToggle:
    def test_toggle_soft_blend(self) -> None:
        renderer = InvisibilityRenderer(RenderingConfig(use_soft_blend=False))
        assert renderer.use_soft_blend is False
        renderer.use_soft_blend = True
        assert renderer.use_soft_blend is True

    def test_toggle_changes_output(self) -> None:
        bg = _solid_bgr(80, 80, (200, 200, 200))
        frame = _solid_bgr(80, 80, (50, 50, 50))
        mask = _mask_rect(80, 80, 20, 60, 20, 60)
        soft = _soft_from_binary(mask, radius=7)

        renderer = InvisibilityRenderer(RenderingConfig(use_soft_blend=False))
        hard_out = renderer.render(frame, bg, mask)

        renderer.use_soft_blend = True
        soft_out = renderer.render(frame, bg, mask, soft_mask=soft)

        # Feathered soft blend creates gradient at edges; hard does not
        # Check an edge pixel — hard has abrupt transition, soft has gradient
        assert hard_out[20, 40, 0] != soft_out[20, 40, 0]


# -- edge cases ---------------------------------------------------------------


class TestEdgeCases:
    def test_1x1_mask(self, hard_renderer: InvisibilityRenderer) -> None:
        bg = _solid_bgr(1, 1, (100, 100, 100))
        frame = _solid_bgr(1, 1, (200, 200, 200))
        mask = np.array([[255]], dtype=np.uint8)
        output = hard_renderer.render(frame, bg, mask)
        assert np.all(output[0, 0] == [100, 100, 100])

    def test_checkerboard_pattern(self, hard_renderer: InvisibilityRenderer) -> None:
        """Alternating masked/unmasked pixels produce correct per-pixel result."""
        h, w = 10, 10
        bg = _solid_bgr(h, w, (255, 0, 0))
        frame = _solid_bgr(h, w, (0, 0, 255))
        mask = np.zeros((h, w), dtype=np.uint8)
        # Set every other pixel
        for y in range(h):
            for x in range(w):
                if (y + x) % 2 == 0:
                    mask[y, x] = 255

        output = hard_renderer.render(frame, bg, mask)

        for y in range(h):
            for x in range(w):
                if (y + x) % 2 == 0:
                    assert np.all(output[y, x] == [255, 0, 0])
                else:
                    assert np.all(output[y, x] == [0, 0, 255])
