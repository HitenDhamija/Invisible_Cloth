"""Tests for ScreenCapturer using synthetic frames (no webcam required)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from cloak.recording.screenshot import ScreenCapturer

# -- helpers -------------------------------------------------------------------


def _fake_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a small synthetic BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = [100, 150, 200]
    return frame


# -- initial state tests -------------------------------------------------------


class TestInitialState:
    def test_count_is_zero(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        assert capturer.count == 0

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "screenshots"
        ScreenCapturer(output_dir=out_dir)
        assert out_dir.is_dir()

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "my_screenshots"
        capturer = ScreenCapturer(output_dir=custom)
        assert capturer._dir == custom
        assert custom.is_dir()


# -- capture tests -------------------------------------------------------------


class TestCapture:
    def test_saves_jpeg_file(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        path = capturer.capture(_fake_frame())

        assert path.exists()
        assert path.suffix == ".jpg"

    def test_increments_count(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        capturer.capture(_fake_frame())
        capturer.capture(_fake_frame())

        assert capturer.count == 2

    def test_returns_path(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        path = capturer.capture(_fake_frame())
        assert isinstance(path, Path)

    def test_file_is_valid_jpeg(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        frame = _fake_frame()
        path = capturer.capture(frame)

        loaded = cv2.imread(str(path))
        assert loaded is not None
        assert loaded.shape == frame.shape

    def test_with_custom_label(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        path = capturer.capture(_fake_frame(), label="render")

        assert "render" in path.name
        assert path.suffix == ".jpg"

    def test_without_label(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        path = capturer.capture(_fake_frame())

        assert path.name.startswith("cloak_")
        # No user label substring like 'render' or 'debug'
        for label in ("render", "debug", "frame"):
            assert label not in path.stem

    def test_with_custom_output_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom"
        capturer = ScreenCapturer(output_dir=custom)
        path = capturer.capture(_fake_frame())

        assert path.parent == custom

    def test_with_custom_quality(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        noisy = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        path_low = capturer.capture(noisy, quality=10)
        time.sleep(1.1)  # ensure different timestamp so file isn't overwritten
        path_high = capturer.capture(noisy, quality=100)

        assert path_low.stat().st_size < path_high.stat().st_size

    def test_preserves_frame_content(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        frame = np.full((32, 32, 3), 42, dtype=np.uint8)
        path = capturer.capture(frame)

        loaded = cv2.imread(str(path))
        np.testing.assert_array_equal(loaded, frame)


# -- capture_pair tests --------------------------------------------------------


class TestCapturePair:
    def test_saves_render_and_debug(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        render = _fake_frame()
        debug = _fake_frame(h=32, w=32)

        paths = capturer.capture_pair(render, debug)

        assert len(paths) == 2
        assert paths[0].exists()
        assert paths[1].exists()
        assert "render" in paths[0].name
        assert "debug" in paths[1].name

    def test_returns_single_path_when_no_debug(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        paths = capturer.capture_pair(_fake_frame())

        assert len(paths) == 1
        assert "render" in paths[0].name

    def test_with_none_debug_frame(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        paths = capturer.capture_pair(_fake_frame(), debug_frame=None)

        assert len(paths) == 1

    def test_increments_count_for_pair(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        capturer.capture_pair(_fake_frame(), _fake_frame())
        assert capturer.count == 2

    def test_increments_count_single(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        capturer.capture_pair(_fake_frame())
        assert capturer.count == 1

    def test_pair_files_are_valid_jpeg(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        render = _fake_frame()
        debug = _fake_frame(h=32, w=32)

        paths = capturer.capture_pair(render, debug)

        for p in paths:
            loaded = cv2.imread(str(p))
            assert loaded is not None

    def test_multiple_pairs(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)

        capturer.capture_pair(_fake_frame(), _fake_frame())
        capturer.capture_pair(_fake_frame(), _fake_frame())

        assert capturer.count == 4


# -- timestamp uniqueness tests ------------------------------------------------


class TestTimestampUniqueness:
    def test_consecutive_captures_have_unique_names(self, tmp_path: Path) -> None:
        capturer = ScreenCapturer(output_dir=tmp_path)
        paths = []
        for _ in range(5):
            paths.append(capturer.capture(_fake_frame()))
            time.sleep(1.1)  # ensure timestamp changes between captures

        names = [p.name for p in paths]
        assert len(set(names)) == 5
