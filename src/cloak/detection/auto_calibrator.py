"""Automatic HSV color calibration using robust statistics.

Replaces manual trackbar-based calibration with an automatic system
that analyzes pixels from a region of interest (ROI) and computes
optimal HSV thresholds.

Algorithm::

    1. Display central ROI rectangle
    2. User places cloak inside ROI, presses C
    3. Collect all pixels from ROI
    4. Convert to HSV
    5. Reject outliers via percentile clipping (2nd/98th)
    6. Compute median and IQR for each channel
    7. Optionally refine with histogram peak analysis
    8. Optionally cluster with K-means (for folds/shadows)
    9. Generate lower/upper HSV bounds with margins
    10. Show preview, user accepts (A) or cancels (X)

Why percentile + IQR over min/max:

    Min/max is maximally sensitive to a single outlier pixel.
    If one bright highlight or one dark shadow pixel exists in the
    ROI, min/max produces terrible bounds. Percentile clipping
    ignores the worst 5% of pixels on each tail, and IQR captures
    the spread of the central bulk of data.

Why median over mean:

    The mean is pulled by outliers. The median is the 50th percentile
    — it is the most robust measure of central tendency for
    non-Gaussian distributions (which real-world cloth colors are,
    due to folds and lighting).

Why K-means (optional):

    A cloth under mixed lighting produces a multimodal HSV distribution
    (bright highlight cluster + mid-tone cluster + shadow cluster).
    Percentile statistics treat this as one wide distribution, which
    may include too much range. K-means separates these clusters, and
    selecting the largest cluster gives a tighter, more accurate range.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import cv2
import numpy as np

from cloak.config.schemas import CalibrationConfig, DetectionConfig

logger = logging.getLogger(__name__)

_WINDOW = "Auto Calibrator"


@dataclasses.dataclass(frozen=True)
class CalibrationResult:
    """Result of automatic HSV calibration."""

    hsv_lower: list[int]
    hsv_upper: list[int]
    method: str
    pixel_count: int
    median_hsv: list[float]
    iqr_hsv: list[float]
    timestamp: str


class CalibrationState:
    """Calibration state machine."""

    INACTIVE = "inactive"
    COLLECTING = "collecting"
    COMPUTING = "computing"
    PREVIEW = "preview"


class AutoCalibrator:
    """Automatic HSV calibration with ROI-based color analysis.

    Example::

        cal = AutoCalibrator(calibration_config, detection_config)
        while True:
            frame = cam.read()
            result, state = cal.step(frame)
            if result is not None:
                detector.set_bounds(result.hsv_lower, result.hsv_upper)
    """

    def __init__(
        self,
        calibration_cfg: CalibrationConfig,
        detection_cfg: DetectionConfig,
    ) -> None:
        self._cfg = calibration_cfg
        self._detection_cfg = detection_cfg
        self._state = CalibrationState.INACTIVE
        self._result: CalibrationResult | None = None
        self._preview_mask: np.ndarray | None = None
        self._collected_pixels: np.ndarray | None = None
        self._roi_rect: tuple[int, int, int, int] = (0, 0, 0, 0)

    # -- public API -----------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def result(self) -> CalibrationResult | None:
        return self._result

    @property
    def preview_mask(self) -> np.ndarray | None:
        return self._preview_mask

    @property
    def is_active(self) -> bool:
        return self._state != CalibrationState.INACTIVE

    def start(self, frame_shape: tuple[int, int, int]) -> None:
        """Begin calibration — compute ROI and enter COLLECTING state."""
        h, w = frame_shape[:2]
        roi_frac = self._cfg.roi_fraction
        roi_h = int(h * roi_frac)
        roi_w = int(w * roi_frac)
        x0 = (w - roi_w) // 2
        y0 = (h - roi_h) // 2
        self._roi_rect = (x0, y0, roi_w, roi_h)
        self._state = CalibrationState.COLLECTING
        self._result = None
        self._preview_mask = None
        self._collected_pixels = None
        logger.info("Calibration started — place cloak in ROI and press C")

    def collect(self, frame: np.ndarray) -> CalibrationResult | None:
        """Collect pixels from ROI and compute calibration.

        Call this when user presses C during COLLECTING state.

        Returns:
            CalibrationResult on success, None if insufficient pixels.
        """
        self._state = CalibrationState.COMPUTING

        # Extract ROI pixels
        x0, y0, rw, rh = self._roi_rect
        roi = frame[y0 : y0 + rh, x0 : x0 + rw]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        pixels = roi_hsv.reshape(-1, 3).astype(np.float32)

        if len(pixels) < self._cfg.min_pixels:
            logger.warning(
                "Insufficient pixels: %d < %d",
                len(pixels), self._cfg.min_pixels,
            )
            self._state = CalibrationState.COLLECTING
            return None

        self._collected_pixels = pixels

        # Compute bounds
        method = "percentile"
        lower, upper, median, iqr = self._compute_percentile_bounds(pixels)

        # Optional histogram refinement
        if self._cfg.use_histogram:
            lower, upper = self._histogram_refine(pixels, lower, upper)
            method = "percentile+histogram"

        # Optional K-means clustering
        if self._cfg.use_kmeans:
            lower, upper, median, iqr = self._kmeans_refine(
                pixels, lower, upper,
            )
            method = "kmeans" if not self._cfg.use_histogram else "kmeans+histogram"

        # Clamp to valid ranges
        lower = [max(0, int(lower[i])) for i in range(3)]
        upper[0] = min(179, int(upper[0]))
        upper[1] = min(255, int(upper[1]))
        upper[2] = min(255, int(upper[2]))

        self._result = CalibrationResult(
            hsv_lower=lower,
            hsv_upper=upper,
            method=method,
            pixel_count=len(pixels),
            median_hsv=[float(median[i]) for i in range(3)],
            iqr_hsv=[float(iqr[i]) for i in range(3)],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Generate preview mask
        self._preview_mask = self._generate_preview(frame)

        self._state = CalibrationState.PREVIEW
        logger.info(
            "Calibration computed: lower=%s upper=%s method=%s pixels=%d",
            lower, upper, method, len(pixels),
        )
        return self._result

    def accept(self) -> tuple[list[int], list[int]]:
        """Accept calibration result.

        Returns:
            Tuple of (lower, upper) HSV bounds.
        """
        if self._result is None:
            raise ValueError("No calibration result to accept")
        lower = self._result.hsv_lower
        upper = self._result.hsv_upper
        self._state = CalibrationState.INACTIVE
        self._result = None
        self._preview_mask = None
        logger.info("Calibration accepted: lower=%s upper=%s", lower, upper)
        return lower, upper

    def cancel(self) -> None:
        """Cancel calibration and return to inactive state."""
        self._state = CalibrationState.INACTIVE
        self._result = None
        self._preview_mask = None
        self._collected_pixels = None
        logger.info("Calibration cancelled")

    def step(self, frame: np.ndarray) -> tuple[str, CalibrationResult | None]:
        """Process one frame of calibration.

        Returns:
            Tuple of (state, result_or_none).
        """
        if self._state == CalibrationState.INACTIVE:
            return self._state, None

        if self._state == CalibrationState.COLLECTING:
            self._draw_roi(frame)

        elif self._state == CalibrationState.PREVIEW:
            self._draw_preview(frame)

        return self._state, self._result

    # -- statistics -----------------------------------------------------------

    def _compute_percentile_bounds(
        self, pixels: np.ndarray,
    ) -> tuple[list[float], list[float], np.ndarray, np.ndarray]:
        """Compute HSV bounds using percentile + IQR.

        Algorithm:
            1. For each HSV channel, compute the p_low and p_high percentiles
            2. The median is the center of the distribution
            3. IQR = Q3 - Q1 measures the spread of the central 50%
            4. Bounds = median +/- (IQR/2 + margin)

        This is robust because:
            - Percentiles ignore outliers entirely
            - IQR is insensitive to extreme values
            - The margin ensures we capture the full distribution width
        """
        p_low = self._cfg.percentile_low
        p_high = self._cfg.percentile_high

        medians = np.median(pixels, axis=0)
        q1 = np.percentile(pixels, 25, axis=0)
        q3 = np.percentile(pixels, 75, axis=0)
        iqr = q3 - q1

        p_low_vals = np.percentile(pixels, p_low, axis=0)
        p_high_vals = np.percentile(pixels, p_high, axis=0)

        margins = np.array([self._cfg.h_margin, self._cfg.s_margin, self._cfg.v_margin])

        lower = p_low_vals - margins
        upper = p_high_vals + margins

        return lower.tolist(), upper.tolist(), medians, iqr

    def _histogram_refine(
        self,
        pixels: np.ndarray,
        lower: list[float],
        upper: list[float],
    ) -> tuple[list[float], list[float]]:
        """Refine bounds using histogram peak analysis.

        For each channel, build a histogram and find the peaks.
        Trim bounds to the region containing the main peak,
        which removes secondary peaks (e.g., background colors
        that leaked into the ROI).
        """
        n_bins = self._cfg.histogram_bins

        for ch in range(3):
            channel_data = pixels[:, ch]
            hist, bin_edges = np.histogram(
                channel_data, bins=n_bins,
                range=(0, 179 if ch == 0 else 255),
            )

            # Find the peak bin
            peak_idx = int(np.argmax(hist))

            # Find the extent where histogram is above 30% of peak
            threshold = hist[peak_idx] * 0.3
            above = np.where(hist >= threshold)[0]

            if len(above) > 0:
                hist_low = bin_edges[above[0]]
                hist_high = bin_edges[above[-1] + 1] if above[-1] + 1 < len(bin_edges) else bin_edges[-1]

                # Widen slightly to be safe
                margin = [self._cfg.h_margin, self._cfg.s_margin, self._cfg.v_margin][ch]
                hist_low = max(0, hist_low - margin)
                hist_high = min(179 if ch == 0 else 255, hist_high + margin)

                # Take the intersection with percentile bounds
                lower[ch] = max(lower[ch], hist_low)
                upper[ch] = min(upper[ch], hist_high)

        # Enforce blue hue constraint: Hue should be in 85-135 range
        # If the computed range doesn't overlap with blue, force it
        if upper[0] < 85 or lower[0] > 135:
            # No blue detected, use default blue range
            lower[0] = 85.0
            upper[0] = 135.0

        # Enforce minimum saturation (blue should have decent saturation)
        if upper[1] < 50:
            upper[1] = 255.0

        return lower, upper

    def _kmeans_refine(
        self,
        pixels: np.ndarray,
        lower: list[float],
        upper: list[float],
    ) -> tuple[list[float], list[float], np.ndarray, np.ndarray]:
        """Refine bounds using K-means clustering.

        What a cluster represents:
            Each cluster is a group of pixels with similar HSV values.
            Under mixed lighting, a single blue cloth produces 2-3 clusters:
            - Bright highlight cluster (high V)
            - Mid-tone cluster (the "true" color)
            - Shadow cluster (low V)

        How the cloak cluster is selected:
            We select the largest cluster (most pixels), which
            corresponds to the dominant surface area of the cloth.
            This is the cluster that matters most for detection.

        Why clustering helps with folds/shadows:
            Without clustering, percentile statistics average across
            all lighting conditions, producing a wide range that may
            include non-cloth colors. Clustering isolates the dominant
            color, producing tighter bounds.
        """
        n_clusters = self._cfg.kmeans_n_clusters

        # Subsample for speed if too many pixels
        max_samples = 5000
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,  # max iterations
            1.0, # epsilon
        )

        # K-means expects float32 data
        sample_f32 = sample.astype(np.float32)

        try:
            compactness, labels, centers = cv2.kmeans(
                sample_f32, n_clusters, None,
                criteria, 10,  # attempts
                cv2.KMEANS_PP_CENTERS,
            )
        except cv2.error:
            # Fallback to percentile bounds if k-means fails
            medians = np.median(pixels, axis=0)
            iqr = np.percentile(pixels, 75, axis=0) - np.percentile(pixels, 25, axis=0)
            return lower, upper, medians, iqr

        # Find the largest cluster
        unique, counts = np.unique(labels, return_counts=True)
        largest_cluster_idx = unique[np.argmax(counts)]
        cluster_mask = (labels.flatten() == largest_cluster_idx)
        cluster_pixels = sample[cluster_mask]

        # Compute bounds from the largest cluster only
        medians = np.median(cluster_pixels, axis=0)
        q1 = np.percentile(cluster_pixels, 25, axis=0)
        q3 = np.percentile(cluster_pixels, 75, axis=0)
        iqr = q3 - q1

        p_low = self._cfg.percentile_low
        p_high = self._cfg.percentile_high
        p_low_vals = np.percentile(cluster_pixels, p_low, axis=0)
        p_high_vals = np.percentile(cluster_pixels, p_high, axis=0)

        margins = np.array([self._cfg.h_margin, self._cfg.s_margin, self._cfg.v_margin])

        lower = (p_low_vals - margins).tolist()
        upper = (p_high_vals + margins).tolist()

        return lower, upper, medians, iqr

    # -- preview ---------------------------------------------------------------

    def _generate_preview(self, frame: np.ndarray) -> np.ndarray:
        """Generate a preview mask using the computed bounds."""
        if self._result is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        lower = np.array(self._result.hsv_lower, dtype=np.uint8)
        upper = np.array(self._result.hsv_upper, dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return mask

    # -- drawing --------------------------------------------------------------

    def _draw_roi(self, frame: np.ndarray) -> None:
        """Draw the ROI rectangle and instructions on the frame."""
        x0, y0, rw, rh = self._roi_rect
        h, w = frame.shape[:2]

        # Semi-transparent overlay outside ROI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Clear the ROI area
        frame[y0 : y0 + rh, x0 : x0 + rw] = \
            frame[y0 : y0 + rh, x0 : x0 + rw].copy()

        # Draw ROI border
        cv2.rectangle(frame, (x0, y0), (x0 + rw, y0 + rh), (0, 255, 255), 2)

        # Instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Place cloak in box, press C"
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = y0 - 15 if y0 > 40 else y0 + rh + 30
        cv2.putText(frame, text, (tx, ty), font, 0.6, (0, 255, 255), 2)

        # Corner markers
        corner_len = 20
        color = (0, 255, 255)
        thickness = 2
        # Top-left
        cv2.line(frame, (x0, y0), (x0 + corner_len, y0), color, thickness)
        cv2.line(frame, (x0, y0), (x0, y0 + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x0 + rw, y0), (x0 + rw - corner_len, y0), color, thickness)
        cv2.line(frame, (x0 + rw, y0), (x0 + rw, y0 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x0, y0 + rh), (x0 + corner_len, y0 + rh), color, thickness)
        cv2.line(frame, (x0, y0 + rh), (x0, y0 + rh - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x0 + rw, y0 + rh), (x0 + rw - corner_len, y0 + rh), color, thickness)
        cv2.line(frame, (x0 + rw, y0 + rh), (x0 + rw, y0 + rh - corner_len), color, thickness)

    def _draw_preview(self, frame: np.ndarray) -> None:
        """Show calibration preview with mask overlay."""
        if self._preview_mask is None or self._result is None:
            return

        h, w = frame.shape[:2]
        preview_w = w // 2
        preview_h = h // 2

        # Original with ROI outline
        orig = frame.copy()
        x0, y0, rw, rh = self._roi_rect
        cv2.rectangle(orig, (x0, y0), (x0 + rw, y0 + rh), (0, 255, 255), 2)

        # Mask visualization
        mask_bgr = cv2.cvtColor(self._preview_mask, cv2.COLOR_GRAY2BGR)

        # Blue region overlay
        blue_region = frame.copy()
        blue_region[self._preview_mask == 0] = 0

        # Combine: original | mask | blue region
        orig_small = cv2.resize(orig, (preview_w, preview_h))
        mask_small = cv2.resize(mask_bgr, (preview_w, preview_h))
        blue_small = cv2.resize(blue_region, (preview_w, preview_h))

        top = np.hstack([orig_small, mask_small])
        bottom = np.hstack([blue_small, np.zeros_like(blue_small)])
        combined = np.vstack([top, bottom])

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 25), font, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, "Preview mask", (preview_w + 10, 25), font, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, "Blue region", (10, preview_h + 25), font, 0.6, (0, 255, 255), 2)

        # Result info
        r = self._result
        info_lines = [
            f"Lower: {r.hsv_lower}  Upper: {r.hsv_upper}",
            f"Method: {r.method}  Pixels: {r.pixel_count}",
            f"Median HSV: [{r.median_hsv[0]:.0f}, {r.median_hsv[1]:.0f}, {r.median_hsv[2]:.0f}]",
            "A: accept  X: cancel",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(
                combined, line, (10, combined.shape[0] - 10 - (len(info_lines) - 1 - i) * 22),
                font, 0.5, (200, 200, 200), 1,
            )

        cv2.imshow(_WINDOW, combined)

    def destroy(self) -> None:
        """Close calibration window."""
        import contextlib
        with contextlib.suppress(cv2.error):
            cv2.destroyWindow(_WINDOW)
