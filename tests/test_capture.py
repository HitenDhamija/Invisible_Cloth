"""Tests for background aggregation functions (no webcam required)."""

from __future__ import annotations

import numpy as np
import pytest

from cloak.capture.aggregator import (
    AggregationError,
    _validate_frames,
    aggregate_mean,
    aggregate_median,
)

# -- helpers -------------------------------------------------------------------


def _solid_frame(shape: tuple[int, int, int], value: int) -> np.ndarray:
    """Create a BGR frame filled with a uniform value."""
    return np.full(shape, value, dtype=np.uint8)


def _gradient_frame(width: int, height: int, channels: int = 3) -> np.ndarray:
    """Create a frame with values ramping from 0..255 across pixels."""
    frame = np.zeros((height, width, channels), dtype=np.uint8)
    for c in range(channels):
        frame[:, :, c] = np.linspace(0, 255, width, dtype=np.uint8)
    return frame


# -- mean aggregation ----------------------------------------------------------


class TestAggregateMean:
    def test_single_frame_passthrough(self) -> None:
        frame = _solid_frame((4, 6, 3), 100)
        result = aggregate_mean([frame])
        np.testing.assert_array_equal(result, frame.astype(np.float64))

    def test_two_identical_frames(self) -> None:
        frame = _solid_frame((4, 6, 3), 80)
        result = aggregate_mean([frame, frame])
        np.testing.assert_array_equal(result, frame.astype(np.float64))

    def test_average_of_two_values(self) -> None:
        f1 = _solid_frame((2, 2, 3), 40)
        f2 = _solid_frame((2, 2, 3), 60)
        result = aggregate_mean([f1, f2])
        expected = np.full((2, 2, 3), 50.0, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_of_three_values(self) -> None:
        frames = [_solid_frame((2, 2, 3), v) for v in [0, 100, 200]]
        result = aggregate_mean(frames)
        expected = np.full((2, 2, 3), 100.0, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_result_is_float64(self) -> None:
        frame = _solid_frame((4, 4, 3), 128)
        result = aggregate_mean([frame, frame])
        assert result.dtype == np.float64

    def test_gradient_averaged_with_self(self) -> None:
        grad = _gradient_frame(100, 50)
        result = aggregate_mean([grad, grad])
        np.testing.assert_array_almost_equal(result, grad.astype(np.float64))


# -- median aggregation --------------------------------------------------------


class TestAggregateMedian:
    def test_single_frame_passthrough(self) -> None:
        frame = _solid_frame((4, 6, 3), 100)
        result = aggregate_median([frame])
        np.testing.assert_array_equal(result, frame.astype(np.float64))

    def test_median_of_identical_frames(self) -> None:
        frame = _solid_frame((4, 6, 3), 80)
        result = aggregate_median([frame, frame, frame])
        np.testing.assert_array_equal(result, frame.astype(np.float64))

    def test_median_suppresses_outlier(self) -> None:
        """A single bright frame among dark frames is suppressed."""
        dark = _solid_frame((4, 4, 3), 20)
        bright = _solid_frame((4, 4, 3), 200)
        # bright appears once out of 5 frames → median should be 20
        result = aggregate_median([dark, dark, bright, dark, dark])
        np.testing.assert_array_equal(result, dark.astype(np.float64))

    def test_median_of_uneven_values(self) -> None:
        frames = [_solid_frame((2, 2, 3), v) for v in [10, 20, 30]]
        result = aggregate_median(frames)
        expected = np.full((2, 2, 3), 20.0, dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_median_across_channels_independently(self) -> None:
        """Each channel is aggregated independently."""
        f1 = np.array([[[10, 200, 50]]], dtype=np.uint8)
        f2 = np.array([[[20, 100, 80]]], dtype=np.uint8)
        f3 = np.array([[[30, 50, 30]]], dtype=np.uint8)
        result = aggregate_median([f1, f2, f3])
        expected = np.array([[[20.0, 100.0, 50.0]]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_result_is_float64(self) -> None:
        frame = _solid_frame((4, 4, 3), 128)
        result = aggregate_median([frame, frame])
        assert result.dtype == np.float64


# -- error handling -----------------------------------------------------------


class TestAggregationErrors:
    def test_empty_list_raises(self) -> None:
        with pytest.raises(AggregationError, match="empty"):
            aggregate_mean([])

    def test_empty_list_median_raises(self) -> None:
        with pytest.raises(AggregationError, match="empty"):
            aggregate_median([])

    def test_single_frame_when_min_is_two(self) -> None:
        frame = _solid_frame((4, 4, 3), 50)
        with pytest.raises(AggregationError, match="at least 2"):
            _validate_frames([frame], min_frames=2)

    def test_mismatched_shapes_raises(self) -> None:
        f1 = _solid_frame((4, 6, 3), 50)
        f2 = _solid_frame((8, 10, 3), 50)
        with pytest.raises(AggregationError, match="shape"):
            aggregate_mean([f1, f2])

    def test_mismatched_shapes_median_raises(self) -> None:
        f1 = _solid_frame((4, 4, 3), 50)
        f2 = _solid_frame((4, 6, 3), 50)
        with pytest.raises(AggregationError, match="shape"):
            aggregate_median([f1, f2])


# -- synthetic foreground suppression test -------------------------------------


class TestForegroundSuppression:
    """Demonstrate that median aggregation suppresses transient objects."""

    def test_person_in_some_frames(self) -> None:
        """Simulate: 9 clean background frames, 1 with a bright 'person' blob."""
        h, w = 48, 64
        bg_value = 30
        person_value = 200

        frames: list[np.ndarray] = []
        for _ in range(9):
            frames.append(_solid_frame((h, w, 3), bg_value))

        # One frame with a bright rectangle in the center (simulating a person)
        person_frame = _solid_frame((h, w, 3), bg_value)
        person_frame[10:40, 20:50] = person_value
        frames.append(person_frame)

        result = aggregate_median(frames)

        # Background pixels outside the person region should be exactly bg_value
        assert result[0, 0, 0] == pytest.approx(bg_value, abs=1.0)

        # The median at the person's location should still be bg_value,
        # because 9 out of 10 frames show background there.
        assert result[25, 35, 0] == pytest.approx(bg_value, abs=1.0)

    def test_mean_does_not_suppress(self) -> None:
        """Mean aggregation will be pulled toward the outlier."""
        h, w = 48, 64
        bg_value = 30
        person_value = 200

        frames: list[np.ndarray] = []
        for _ in range(9):
            frames.append(_solid_frame((h, w, 3), bg_value))

        person_frame = _solid_frame((h, w, 3), bg_value)
        person_frame[10:40, 20:50] = person_value
        frames.append(person_frame)

        result = aggregate_mean(frames)

        # The mean at the person's location will be pulled above bg_value
        # (bg=30, person=200, 9 clean + 1 dirty → mean ≈ 47)
        assert result[25, 35, 0] > bg_value + 5
