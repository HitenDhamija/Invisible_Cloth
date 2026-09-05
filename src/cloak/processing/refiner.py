"""Mask refinement pipeline.

Converts a noisy raw binary mask from HSV detection into a clean,
visually stable cloak mask suitable for compositing.

Pipeline order and rationale::

    1. Median blur         — removes salt-and-pepper noise (isolated
                              flickering pixels) without blurring edges
                              as aggressively as a Gaussian.
    2. Morphological OPEN  — erode then dilate: removes small isolated
                              foreground blobs (tiny blue objects in the
                              room) while preserving large regions.
    3. Morphological CLOSE — dilate then erode: fills small holes inside
                              the detected cloth (folds, wrinkles where
                              saturation briefly drops).
    4. Dilation            — expands the mask boundary outward to ensure
                              full cloth coverage (avoids thin un-masked
                              borders).
    5. Erosion             — shrinks the mask back, but since dilation
                              and erosion use the same kernel the net
                              effect is boundary smoothing.  If dilation
                              > erosion, the mask stays slightly膨胀
                              for better coverage.
    6. Contour filtering   — removes connected components smaller than
                              a configurable area threshold, rejecting
                              small blue objects (e.g. a pen, a book
                              spine) that should not become invisible.
    7. Soft mask           — optional Gaussian-feathered version of the
                              binary mask for smooth alpha blending at
                              cloth edges in future compositing.

Each step is independently configurable so the pipeline can be tuned
for different cloth sizes, lighting conditions, and camera resolutions.
"""

from __future__ import annotations

import dataclasses
import logging

import cv2
import numpy as np

from cloak.config.schemas import MaskConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RefinementStats:
    """Statistics produced by the mask refinement pipeline."""

    raw_pixel_count: int
    refined_pixel_count: int
    components_total: int
    components_rejected: int
    soft_mask_generated: bool


class MaskRefiner:
    """Refine a raw binary mask into a clean cloak mask.

    Example::

        refiner = MaskRefiner(config.mask)
        binary, soft, stats = refiner.refine(raw_mask)
    """

    def __init__(self, config: MaskConfig) -> None:
        self._cfg = config
        self._kernel = self._build_kernel(config.morphology_kernel)
        self._stats = RefinementStats(0, 0, 0, 0, False)

    @property
    def last_stats(self) -> RefinementStats:
        return self._stats

    # -- main entry point -----------------------------------------------------

    def refine(self, raw_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, RefinementStats]:
        """Run the full refinement pipeline.

        Args:
            raw_mask: Binary uint8 mask (0 or 255) from the detector.

        Returns:
            Tuple of (binary_mask, soft_mask_or_None, stats).
            *binary_mask* is uint8 with values 0 or 255.
            *soft_mask* is float32 in [0.0, 1.0] with feathered edges,
            or *None* if feather_radius is 0.
        """
        raw_count = int(cv2.countNonZero(raw_mask))
        mask = raw_mask.copy()

        # 1. Median blur — salt-and-pepper removal
        mask = self._median_blur(mask)

        # 2. Morphological open — remove small noise blobs
        mask = self._open(mask)

        # 3. Morphological close — fill small holes
        mask = self._close(mask)

        # 4–5. Dilation then erosion — boundary smoothing / expansion
        mask = self._dilate(mask)
        mask = self._erode(mask)

        # 6. Contour filtering — reject small components
        mask, components_total, components_rejected = self._filter_components(mask)

        refined_count = int(cv2.countNonZero(mask))

        # 7. Soft mask (feathered edges)
        soft: np.ndarray | None = None
        if self._cfg.feather_radius > 0:
            soft = self._feather(mask)

        self._stats = RefinementStats(
            raw_pixel_count=raw_count,
            refined_pixel_count=refined_count,
            components_total=components_total,
            components_rejected=components_rejected,
            soft_mask_generated=soft is not None,
        )

        logger.debug(
            "Refinement: %d → %d pixels (%d/%d components kept)",
            raw_count,
            refined_count,
            components_total - components_rejected,
            components_total,
        )

        return mask, soft, self._stats

    # -- pipeline stages ------------------------------------------------------

    def _median_blur(self, mask: np.ndarray) -> np.ndarray:
        k = self._cfg.median_kernel
        if k <= 1:
            return mask
        k = k if k % 2 == 1 else k + 1
        return cv2.medianBlur(mask, k)

    def _open(self, mask: np.ndarray) -> np.ndarray:
        iters = self._cfg.open_iterations
        if iters <= 0:
            return mask
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=iters)

    def _close(self, mask: np.ndarray) -> np.ndarray:
        iters = self._cfg.close_iterations
        if iters <= 0:
            return mask
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=iters)

    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        iters = self._cfg.dilation_iterations
        if iters <= 0:
            return mask
        return cv2.dilate(mask, self._kernel, iterations=iters)

    def _erode(self, mask: np.ndarray) -> np.ndarray:
        iters = self._cfg.erosion_iterations
        if iters <= 0:
            return mask
        return cv2.erode(mask, self._kernel, iterations=iters)

    def _filter_components(self, mask: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Remove connected components smaller than min_region_area.

        Returns:
            (filtered_mask, total_components, rejected_components)
        """
        min_area = self._cfg.min_region_area
        if min_area <= 0:
            return mask, 0, 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Label 0 is always the background
        total = num_labels - 1
        rejected = 0
        result = np.zeros_like(mask)

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= min_area:
                result[labels == label_id] = 255
            else:
                rejected += 1

        return result, total, rejected

    def _feather(self, mask: np.ndarray) -> np.ndarray:
        """Create a soft mask with Gaussian-feathered edges.

        Returns:
            Float32 array in [0.0, 1.0].
        """
        r = self._cfg.feather_radius
        if r <= 0:
            return mask.astype(np.float32) / 255.0

        k = r * 2 + 1  # ensure odd kernel
        # Normalize binary mask to [0, 1]
        binary = mask.astype(np.float32) / 255.0
        # Gaussian blur creates smooth gradient at boundaries
        soft = cv2.GaussianBlur(binary, (k, k), 0)
        return np.clip(soft, 0.0, 1.0)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _build_kernel(size: int) -> np.ndarray:
        k = max(1, size)
        k = k if k % 2 == 1 else k + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
