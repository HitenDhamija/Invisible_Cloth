"""Tests for mask refinement using synthetic masks (no webcam required)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cloak.config.schemas import MaskConfig
from cloak.processing.refiner import MaskRefiner

# -- helpers -------------------------------------------------------------------


def _empty_mask(h: int = 100, w: int = 100) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _full_mask(h: int = 100, w: int = 100) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _rect_mask(
    h: int, w: int,
    y0: int, y1: int, x0: int, x1: int,
) -> np.ndarray:
    mask = _empty_mask(h, w)
    mask[y0:y1, x0:x1] = 255
    return mask


def _noisy_mask(
    h: int, w: int,
    region: tuple[int, int, int, int] | None = None,
    noise_count: int = 50,
    rng_seed: int = 42,
) -> np.ndarray:
    """Mask with a large region plus scattered single-pixel noise."""
    mask = _empty_mask(h, w)
    if region is not None:
        y0, y1, x0, x1 = region
        mask[y0:y1, x0:x1] = 255
    rng = np.random.default_rng(rng_seed)
    ys = rng.integers(0, h, size=noise_count)
    xs = rng.integers(0, w, size=noise_count)
    mask[ys, xs] = 255
    return mask


def _mask_with_hole(
    h: int = 100, w: int = 100,
    hole: tuple[int, int, int, int] = (40, 60, 40, 60),
) -> np.ndarray:
    """Large filled region with a rectangular hole in the center."""
    mask = _full_mask(h, w)
    y0, y1, x0, x1 = hole
    mask[y0:y1, x0:x1] = 0
    return mask


def _multi_region_mask(
    h: int = 200, w: int = 200,
) -> tuple[np.ndarray, int]:
    """One large region + several small regions. Returns (mask, expected_large)."""
    mask = _empty_mask(h, w)
    # Large region: 100x100 = 10000 px
    mask[10:110, 10:110] = 255
    # Small regions: 10x10 = 100 px each
    for i in range(5):
        y = 10 + i * 20
        mask[y:y + 10, 150:160] = 255
    return mask, 1


# -- config fixture -----------------------------------------------------------


@pytest.fixture
def default_mask_cfg() -> MaskConfig:
    return MaskConfig(
        median_kernel=5,
        morphology_kernel=5,
        open_iterations=1,
        close_iterations=2,
        dilation_iterations=1,
        erosion_iterations=1,
        min_region_area=500,
        feather_radius=0,
    )


@pytest.fixture
def refiner(default_mask_cfg: MaskConfig) -> MaskRefiner:
    return MaskRefiner(default_mask_cfg)


# -- basic pipeline tests -----------------------------------------------------


class TestRefinerBasics:
    def test_empty_mask_stays_empty(self, refiner: MaskRefiner) -> None:
        mask = _empty_mask()
        binary, soft, stats = refiner.refine(mask)
        assert cv2.countNonZero(binary) == 0
        assert stats.refined_pixel_count == 0

    def test_full_mask_stays_full(self, refiner: MaskRefiner) -> None:
        mask = _full_mask()
        binary, _soft, stats = refiner.refine(mask)
        assert cv2.countNonZero(binary) > 0
        assert stats.raw_pixel_count == 10000

    def test_output_is_uint8(self, refiner: MaskRefiner) -> None:
        mask = _rect_mask(80, 80, 10, 70, 10, 70)
        binary, _, _ = refiner.refine(mask)
        assert binary.dtype == np.uint8

    def test_output_is_binary(self, refiner: MaskRefiner) -> None:
        mask = _rect_mask(80, 80, 10, 70, 10, 70)
        binary, _, _ = refiner.refine(mask)
        unique = np.unique(binary)
        assert set(unique.tolist()).issubset({0, 255})


# -- noise removal tests ------------------------------------------------------


class TestNoiseRemoval:
    def test_isolated_noise_removed(self, refiner: MaskRefiner) -> None:
        """Single-pixel noise should be eliminated by open + median blur."""
        mask = _noisy_mask(200, 200, region=None, noise_count=100)
        binary, _, stats = refiner.refine(mask)
        # All noise should be gone — either by morphological open or
        # by component filtering (each blob is 1 px < min_region_area)
        assert cv2.countNonZero(binary) == 0
        assert stats.refined_pixel_count == 0

    def test_large_region_preserved(self, refiner: MaskRefiner) -> None:
        """A large solid region should survive refinement."""
        mask = _noisy_mask(200, 200, region=(20, 120, 20, 120), noise_count=30)
        binary, _, stats = refiner.refine(mask)
        # The 100x100 region should still exist
        assert cv2.countNonZero(binary) > 5000
        assert stats.refined_pixel_count > 0

    def test_noise_removed_large_kept(self, refiner: MaskRefiner) -> None:
        """Noise pixels removed, large region kept."""
        mask = _noisy_mask(200, 200, region=(20, 120, 20, 120), noise_count=50)
        binary, _, _ = refiner.refine(mask)
        inner = binary[40:100, 40:100]
        assert np.all(inner == 255)


# -- hole filling tests -------------------------------------------------------


class TestHoleFilling:
    def test_close_fills_small_holes(self) -> None:
        """Morphological close should fill small holes inside the region."""
        cfg = MaskConfig(
            median_kernel=1,
            morphology_kernel=7,
            open_iterations=0,
            close_iterations=3,
            dilation_iterations=0,
            erosion_iterations=0,
            min_region_area=0,
            feather_radius=0,
        )
        refiner = MaskRefiner(cfg)

        # 100x100 mask with a 6x6 hole
        mask = _mask_with_hole(100, 100, hole=(47, 53, 47, 53))
        binary, _, _ = refiner.refine(mask)

        # The small hole should be mostly filled
        center = binary[48:52, 48:52]
        filled_ratio = np.sum(center == 255) / center.size
        assert filled_ratio > 0.8


# -- boundary expansion tests -------------------------------------------------


class TestBoundaryExpansion:
    def test_dilation_expands_mask(self) -> None:
        """Dilation alone should expand the mask."""
        cfg = MaskConfig(
            median_kernel=1,
            morphology_kernel=5,
            open_iterations=0,
            close_iterations=0,
            dilation_iterations=3,
            erosion_iterations=0,
            min_region_area=0,
            feather_radius=0,
        )
        refiner = MaskRefiner(cfg)

        mask = _rect_mask(100, 100, 30, 70, 30, 70)
        raw_count = cv2.countNonZero(mask)

        binary, _, _ = refiner.refine(mask)
        refined_count = cv2.countNonZero(binary)

        assert refined_count > raw_count


# -- contour / component filtering tests --------------------------------------


class TestComponentFiltering:
    def test_small_components_rejected(self) -> None:
        mask, num_large = _multi_region_mask(200, 200)
        cfg = MaskConfig(
            median_kernel=1,
            morphology_kernel=3,
            open_iterations=0,
            close_iterations=0,
            dilation_iterations=0,
            erosion_iterations=0,
            min_region_area=500,
            feather_radius=0,
        )
        refiner = MaskRefiner(cfg)
        binary, _, stats = refiner.refine(mask)

        # The 5 small 10x10 regions (100 px each) should be rejected
        assert stats.components_rejected >= 4
        # The large region should survive
        assert cv2.countNonZero(binary) > 5000

    def test_min_area_zero_disables_filtering(self) -> None:
        mask, _ = _multi_region_mask(200, 200)
        cfg = MaskConfig(min_region_area=0)
        refiner = MaskRefiner(cfg)
        binary, _, stats = refiner.refine(mask)
        # Nothing should be rejected
        assert stats.components_rejected == 0

    def test_all_small_rejected(self) -> None:
        """If all regions are below threshold, mask becomes empty."""
        mask = _empty_mask(100, 100)
        # Five 5x5 regions = 25 px each
        for i in range(5):
            y = 10 + i * 15
            mask[y:y + 5, 10:15] = 255

        cfg = MaskConfig(
            median_kernel=1,
            morphology_kernel=3,
            open_iterations=0,
            close_iterations=0,
            dilation_iterations=0,
            erosion_iterations=0,
            min_region_area=100,
            feather_radius=0,
        )
        refiner = MaskRefiner(cfg)
        binary, _, stats = refiner.refine(mask)
        assert cv2.countNonZero(binary) == 0
        assert stats.components_rejected == 5


# -- soft mask tests ----------------------------------------------------------


class TestSoftMask:
    def test_soft_mask_generated_when_enabled(self) -> None:
        cfg = MaskConfig(feather_radius=7)
        refiner = MaskRefiner(cfg)
        mask = _rect_mask(100, 100, 20, 80, 20, 80)
        _binary, soft, stats = refiner.refine(mask)
        assert soft is not None
        assert stats.soft_mask_generated is True

    def test_soft_mask_not_generated_when_disabled(self) -> None:
        cfg = MaskConfig(feather_radius=0)
        refiner = MaskRefiner(cfg)
        mask = _rect_mask(100, 100, 20, 80, 20, 80)
        _binary, soft, stats = refiner.refine(mask)
        assert soft is None
        assert stats.soft_mask_generated is False

    def test_soft_mask_is_float32(self) -> None:
        cfg = MaskConfig(feather_radius=5)
        refiner = MaskRefiner(cfg)
        mask = _full_mask(60, 60)
        _binary, soft, _ = refiner.refine(mask)
        assert soft is not None
        assert soft.dtype == np.float32

    def test_soft_mask_range_0_to_1(self) -> None:
        cfg = MaskConfig(feather_radius=5)
        refiner = MaskRefiner(cfg)
        mask = _rect_mask(80, 80, 20, 60, 20, 60)
        _binary, soft, _ = refiner.refine(mask)
        assert soft is not None
        assert float(np.min(soft)) >= 0.0
        assert float(np.max(soft)) <= 1.0

    def test_soft_mask_has_gradient_at_edges(self) -> None:
        """The soft mask should have values between 0 and 1 at boundaries."""
        cfg = MaskConfig(feather_radius=10)
        refiner = MaskRefiner(cfg)
        mask = _rect_mask(100, 100, 30, 70, 30, 70)
        _binary, soft, _ = refiner.refine(mask)
        assert soft is not None
        # Check a band around the edge for intermediate values
        edge_band = soft[25:35, 30:70]
        has_intermediate = np.any((edge_band > 0.05) & (edge_band < 0.95))
        assert has_intermediate


# -- statistics tests ---------------------------------------------------------


class TestRefinementStats:
    def test_raw_count_matches_input(self, refiner: MaskRefiner) -> None:
        mask = _rect_mask(100, 100, 10, 50, 10, 90)
        _, _, stats = refiner.refine(mask)
        assert stats.raw_pixel_count == cv2.countNonZero(mask)

    def test_refined_count_matches_output(self, refiner: MaskRefiner) -> None:
        mask = _rect_mask(100, 100, 10, 50, 10, 90)
        binary, _, stats = refiner.refine(mask)
        assert stats.refined_pixel_count == cv2.countNonZero(binary)

    def test_components_total_non_negative(self, refiner: MaskRefiner) -> None:
        mask = _noisy_mask(100, 100, region=(20, 80, 20, 80))
        _, _, stats = refiner.refine(mask)
        assert stats.components_total >= 0

    def test_rejected_leq_total(self, refiner: MaskRefiner) -> None:
        mask = _multi_region_mask(200, 200)[0]
        _, _, stats = refiner.refine(mask)
        assert stats.components_rejected <= stats.components_total


# -- edge cases ---------------------------------------------------------------


class TestEdgeCases:
    def test_1x1_mask(self) -> None:
        cfg = MaskConfig(median_kernel=1, morphology_kernel=1)
        refiner = MaskRefiner(cfg)
        mask = np.array([[255]], dtype=np.uint8)
        binary, _, _ = refiner.refine(mask)
        assert binary.shape == (1, 1)

    def test_all_iterations_zero(self) -> None:
        cfg = MaskConfig(
            median_kernel=1,
            morphology_kernel=3,
            open_iterations=0,
            close_iterations=0,
            dilation_iterations=0,
            erosion_iterations=0,
            min_region_area=0,
            feather_radius=0,
        )
        refiner = MaskRefiner(cfg)
        mask = _rect_mask(60, 60, 10, 50, 10, 50)
        binary, _, _ = refiner.refine(mask)
        # Without any operations, mask should be nearly identical
        assert cv2.countNonZero(binary) == cv2.countNonZero(mask)
