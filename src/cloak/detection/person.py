"""Person detection using MediaPipe Pose Landmarker.

This module performs lazy imports of mediapipe to avoid
loading the AI dependency when person-aware mode is disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cloak.config.schemas import AIConfig

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
_MODEL_NAME = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


class PersonDetectorError(Exception):
    """Raised when MediaPipe fails to initialize or process a frame."""


class PersonDetector:
    """Detect person regions using MediaPipe Pose Landmarker.

    Example::

        detector = PersonDetector(config.ai)
        person_mask = detector.detect(frame, timestamp_ms=0)
        # person_mask is float32 [0.0, 1.0] per pixel
    """

    def __init__(self, config: AIConfig) -> None:
        self._cfg = config
        self._landmarker = None
        self._last_person_mask: np.ndarray | None = None
        self._available = False

    def _ensure_initialized(self) -> None:
        """Lazy-initialize MediaPipe on first use."""
        if self._landmarker is not None:
            return

        try:
            import mediapipe as mp  # noqa: F401
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise PersonDetectorError(
                "mediapipe is required for person-aware mode. "
                "Install it with: pip install mediapipe"
            ) from exc

        model_name = _MODEL_NAME.get(
            self._cfg.model_complexity, "pose_landmarker_lite.task"
        )
        model_path = _MODEL_DIR / model_name

        if not model_path.exists():
            self._download_model(model_name, model_path)

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=True,
            min_pose_detection_confidence=self._cfg.min_detection_confidence,
            min_tracking_confidence=self._cfg.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._available = True
        logger.info("Person detector initialized (model: %s)", model_name)

    @staticmethod
    def _download_model(model_name: str, dest: Path) -> None:
        """Download the pose landmarker model if not present."""
        import urllib.request

        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            f"pose_landmarker/{model_name}/float16/latest/{model_name}"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading pose model: %s", model_name)
        urllib.request.urlretrieve(url, dest)
        logger.info("Model saved to: %s", dest)

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, frame: np.ndarray, timestamp_ms: int = 0) -> np.ndarray:
        """Detect person region in frame.

        Args:
            frame: BGR frame (uint8, H x W x 3).
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            Person segmentation mask (float32, H x W, values 0.0-1.0).
            Returns all-zeros mask if no person detected or on error.
        """
        h, w = frame.shape[:2]
        zeros = np.zeros((h, w), dtype=np.float32)

        if not self._cfg.enabled:
            return zeros

        try:
            self._ensure_initialized()
        except PersonDetectorError:
            return zeros

        try:
            import mediapipe as mp

            rgb = frame[:, :, ::-1].copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.segmentation_masks:
                mask = result.segmentation_masks[0].numpy_view()
                self._last_person_mask = mask.astype(np.float32)
                return self._last_person_mask

            return zeros

        except Exception as exc:
            logger.warning("Person detection failed: %s", exc)
            if self._last_person_mask is not None:
                return self._last_person_mask
            return zeros

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.debug("Person detector closed")
