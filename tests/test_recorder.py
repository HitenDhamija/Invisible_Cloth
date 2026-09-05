"""Tests for VideoRecorder using synthetic frames (no webcam required)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cloak.recording.recorder import RecorderError, VideoRecorder

# -- helpers -------------------------------------------------------------------


def _fake_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a small synthetic BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_mock_writer() -> MagicMock:
    """Create a mock cv2.VideoWriter that reports isOpened=True."""
    writer = MagicMock()
    writer.isOpened.return_value = True
    return writer


# -- initial state tests -------------------------------------------------------


class TestInitialState:
    def test_not_recording(self) -> None:
        rec = VideoRecorder()
        assert rec.is_recording is False

    def test_frame_count_zero(self) -> None:
        rec = VideoRecorder()
        assert rec.frame_count == 0

    def test_duration_zero(self) -> None:
        rec = VideoRecorder()
        assert rec.duration_seconds == 0.0

    def test_current_file_none(self) -> None:
        rec = VideoRecorder()
        assert rec.current_file is None

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        assert rec._dir == tmp_path


# -- start tests ---------------------------------------------------------------


class TestStart:
    def test_creates_output_directory(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "videos"
        rec = VideoRecorder(output_dir=out_dir)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480, 30.0)

        assert out_dir.is_dir()

    def test_returns_path(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            result = rec.start(640, 480)

        assert isinstance(result, Path)
        assert result.parent == tmp_path
        assert result.suffix == ".mp4"

    def test_sets_recording_state(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)

        assert rec.is_recording is True
        assert rec.current_file is not None

    def test_with_custom_output_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "my_videos"
        rec = VideoRecorder(output_dir=custom)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            path = rec.start(640, 480)

        assert path.parent == custom
        assert custom.is_dir()


# -- write tests ---------------------------------------------------------------


class TestWrite:
    def test_returns_true_when_recording(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            result = rec.write(_fake_frame())

        assert result is True

    def test_increments_frame_count(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            rec.write(_fake_frame())
            rec.write(_fake_frame())
            rec.write(_fake_frame())

        assert rec.frame_count == 3

    def test_returns_false_when_not_recording(self) -> None:
        rec = VideoRecorder()
        result = rec.write(_fake_frame())
        assert result is False

    def test_calls_writer_write(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            frame = _fake_frame()
            rec.write(frame)

        mock_writer.write.assert_called_once_with(frame)


# -- stop tests ----------------------------------------------------------------


class TestStop:
    def test_returns_path(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            result = rec.stop()

        assert isinstance(result, Path)

    def test_sets_not_recording(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            rec.stop()

        assert rec.is_recording is False

    def test_returns_none_when_not_recording(self) -> None:
        rec = VideoRecorder()
        assert rec.stop() is None

    def test_releases_writer(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            rec.stop()

        mock_writer.release.assert_called_once()
        assert rec._writer is None


# -- double-start tests --------------------------------------------------------


class TestDoubleStart:
    def test_raises_recorder_error(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            with pytest.raises(RecorderError, match="Already recording"):
                rec.start(640, 480)


# -- close tests ---------------------------------------------------------------


class TestClose:
    def test_close_is_alias_for_stop(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            rec.close()

        assert rec.is_recording is False
        mock_writer.release.assert_called_once()

    def test_close_when_not_recording(self) -> None:
        rec = VideoRecorder()
        rec.close()
        assert rec.is_recording is False


# -- duration tests ------------------------------------------------------------


class TestDuration:
    def test_duration_updates_while_recording(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            time.sleep(0.05)
            d1 = rec.duration_seconds
            time.sleep(0.05)
            d2 = rec.duration_seconds

        assert d2 > d1 > 0.0

    def test_duration_zero_after_stop(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            time.sleep(0.02)
            rec.stop()

        assert rec.duration_seconds == 0.0


# -- error handling tests ------------------------------------------------------


class TestErrorHandling:
    def test_writer_not_opened_raises(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        bad_writer = MagicMock()
        bad_writer.isOpened.return_value = False

        with (
            patch("cloak.recording.recorder.cv2.VideoWriter", return_value=bad_writer),
            pytest.raises(RecorderError, match="Could not open"),
        ):
            rec.start(640, 480)

    def test_write_exception_returns_false(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()
        mock_writer.write.side_effect = OSError("disk full")

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            result = rec.write(_fake_frame())

        assert result is False
        assert rec.frame_count == 0


# -- multiple session tests ----------------------------------------------------


class TestMultipleSessions:
    def test_start_stop_start(self, tmp_path: Path) -> None:
        rec = VideoRecorder(output_dir=tmp_path)
        mock_writer = _make_mock_writer()

        with patch("cloak.recording.recorder.cv2.VideoWriter", return_value=mock_writer):
            rec.start(640, 480)
            rec.write(_fake_frame())
            path1 = rec.stop()

            time.sleep(1.1)  # ensure timestamp changes

            rec.start(640, 480)
            rec.write(_fake_frame())
            rec.write(_fake_frame())
            path2 = rec.stop()

        assert path1 != path2
        assert rec.frame_count == 2
