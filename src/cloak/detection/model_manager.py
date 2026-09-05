"""AI model loading and management for person segmentation.

Provides lazy-loading, device detection, and clean error handling for
ONNX Runtime inference. Supports both CPU and GPU execution providers.

Architecture::

    ModelManager
    ├── lazy_load()        -- loads model on first inference
    ├── predict()          -- runs inference with input preprocessing
    ├── device_info        -- reports CPU/GPU availability
    └── close()            -- releases session resources

The manager does not assume any specific model architecture. It accepts
any valid ONNX model and processes its outputs generically.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from cloak.config.schemas import AIConfig

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"


class ModelManagerError(Exception):
    """Raised when the model fails to load or process frames."""


class ModelManager:
    """Lazy-loading ONNX Runtime model manager for person segmentation.

    Example::

        manager = ModelManager(config.ai)
        mask = manager.predict(frame)
        print(manager.last_latency_ms)
    """

    def __init__(self, config: AIConfig) -> None:
        self._cfg = config
        self._session = None
        self._input_name: str = ""
        self._input_shape: list[int] = []
        self._loaded = False
        self._last_latency_ms: float = 0.0
        self._frame_count: int = 0

        # Inference resolution from config
        self._infer_w = config.inference_width
        self._infer_h = config.inference_height

    # -- public API -----------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if onnxruntime is importable and model is loaded."""
        return self._loaded

    @property
    def last_latency_ms(self) -> float:
        """Latency of the most recent predict() call in milliseconds."""
        return self._last_latency_ms

    @property
    def frame_count(self) -> int:
        """Total frames processed since last load."""
        return self._frame_count

    @property
    def device_info(self) -> str:
        """Human-readable device string (e.g. 'CPU', 'CUDA', 'TensorRT')."""
        if not self._loaded:
            return "not loaded"
        providers = self._session.get_providers()  # type: ignore[union-attr]
        if "CUDAExecutionProvider" in providers:
            return "GPU (CUDA)"
        if "TensorrtExecutionProvider" in providers:
            return "GPU (TensorRT)"
        return "CPU"

    def ensure_loaded(self) -> None:
        """Lazy-load the model on first use.

        Raises:
            ModelManagerError: If onnxruntime is missing or model file
                cannot be found/loaded.
        """
        if self._loaded:
            return

        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as exc:
            raise ModelManagerError(
                "onnxruntime is required for AI hybrid mode. "
                "Install it with: pip install onnxruntime"
            ) from exc

        model_path = self._resolve_model_path()
        self._create_session(ort, model_path)
        self._loaded = True
        logger.info(
            "Model loaded: %s (device: %s, input: %s)",
            model_path.name,
            self.device_info,
            self._input_shape,
        )

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Run person segmentation on a BGR frame.

        Args:
            frame: BGR input image (uint8, H x W x 3).

        Returns:
            Binary person mask (uint8, H x W, values 0 or 255).

        Raises:
            ModelManagerError: If model is not loaded or inference fails.
        """
        self.ensure_loaded()

        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        # Preprocess: resize + normalize to NCHW float32
        resized = cv2.resize(frame, (self._infer_w, self._infer_h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # add batch dim

        # Run inference
        try:
            outputs = self._session.run(None, {self._input_name: blob})  # type: ignore[union-attr]
        except Exception as exc:
            raise ModelManagerError(f"Inference failed: {exc}") from exc

        # Postprocess: extract person mask from model output
        mask = self._postprocess(outputs, h, w)

        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._frame_count += 1

        return mask

    def close(self) -> None:
        """Release ONNX Runtime session resources."""
        if self._session is not None:
            self._session = None
            self._loaded = False
            self._frame_count = 0
            logger.debug("Model manager closed")

    # -- internals ------------------------------------------------------------

    def _resolve_model_path(self) -> Path:
        """Find the model file from config or auto-download."""
        cfg_path = self._cfg.hybrid_model_path.strip()
        if cfg_path:
            path = Path(cfg_path)
            if not path.exists():
                raise ModelManagerError(f"Model file not found: {path}")
            return path

        # Try default locations
        default_name = "yolov8n-seg.onnx"
        candidates = [
            _DEFAULT_MODEL_DIR / default_name,
            Path.cwd() / "models" / default_name,
            Path.cwd() / default_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Auto-download
        dest = _DEFAULT_MODEL_DIR / default_name
        self._download_model(default_name, dest)
        return dest

    @staticmethod
    def _download_model(model_name: str, dest: Path) -> None:
        """Download YOLOv8n-seg ONNX model if not present."""
        import urllib.request

        url = (
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/"
            f"{model_name}"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading segmentation model: %s", model_name)
        try:
            urllib.request.urlretrieve(url, dest)
            logger.info("Model saved to: %s", dest)
        except Exception as exc:
            raise ModelManagerError(
                f"Failed to download model: {exc}. "
                f"Manually download {model_name} and set hybrid_model_path in config."
            ) from exc

    def _create_session(self, ort: object, model_path: Path) -> None:
        """Create ONNX Runtime session with device selection."""
        # Determine available providers
        available_providers = ort.get_available_providers()
        logger.debug("Available providers: %s", available_providers)

        providers: list[str | dict] = []

        # Prefer CUDA if available and not restricted to CPU
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
        else:
            providers.append("CPUExecutionProvider")

        # Create session options
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 2

        try:
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_opts,
                providers=providers,
            )
        except Exception as exc:
            raise ModelManagerError(
                f"Failed to create ONNX session: {exc}"
            ) from exc

        # Read input metadata
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        logger.debug(
            "Session created: input=%s shape=%s providers=%s",
            self._input_name,
            self._input_shape,
            self._session.get_providers(),
        )

    def _postprocess(
        self, outputs: list[np.ndarray], target_h: int, target_w: int,
    ) -> np.ndarray:
        """Convert raw model output to a binary person mask.

        Handles two common output formats:
        1. YOLOv8-seg: outputs[0] is (1, num_classes+4, num_anchors)
           and outputs[1] is prototype masks — we use a simplified
           approach extracting the person class (class 0 in COCO).
        2. Generic segmentation: outputs[0] is (1, 1, H, W) or (1, H, W)
           probability map — threshold directly.
        """
        raw = outputs[0]

        if raw.ndim == 4:
            # Shape (1, C, H, W) — semantic segmentation output
            # Take class 0 (person) or max along class axis
            prob = raw[0, 0] if raw.shape[1] >= 1 else raw[0].max(axis=0)
        elif raw.ndim == 3:
            # Shape (1, N, M) — could be detection output
            # For YOLO: try to extract person-class detections
            prob = self._parse_yolo_output(raw, target_h, target_w)
        elif raw.ndim == 2:
            # Shape (H, W) — single mask
            prob = raw
        else:
            # Fallback: flatten and reshape
            prob = raw.flatten()[: target_h * target_w].reshape(target_h, target_w)

        # Ensure float and in range [0, 1]
        if prob.dtype != np.float32:
            prob = prob.astype(np.float32)
        prob = np.clip(prob, 0.0, 1.0)

        # Resize to target dimensions if needed
        if prob.shape != (target_h, target_w):
            prob = cv2.resize(prob, (target_w, target_h))

        # Threshold to binary mask
        binary = (prob >= self._cfg.confidence_threshold).astype(np.uint8) * 255
        return binary

    def _parse_yolo_output(
        self, raw: np.ndarray, target_h: int, target_w: int,
    ) -> np.ndarray:
        """Parse YOLOv8 segmentation output into a person mask.

        YOLOv8-seg outputs shape (1, 4+nc, num_anchors) where:
        - First 4 channels are box coords (cx, cy, w, h)
        - Remaining nc channels are class confidence scores

        For person detection, we extract anchors where class 0 (person)
        confidence exceeds the threshold, then produce a simple
        spatial activation map.
        """
        data = raw[0]  # remove batch dim -> (4+nc, num_anchors)
        num_anchors = data.shape[1]

        if num_anchors == 0:
            return np.zeros((target_h, target_w), dtype=np.float32)

        # Extract box coordinates and class scores
        boxes = data[:4].T  # (num_anchors, 4)
        class_scores = data[4:].T  # (num_anchors, nc)

        # Person class = index 0 in COCO
        if class_scores.shape[1] < 1:
            return np.zeros((target_h, target_w), dtype=np.float32)

        person_scores = class_scores[:, 0]

        # Filter by confidence
        mask = person_scores >= self._cfg.confidence_threshold
        if not mask.any():
            return np.zeros((target_h, target_w), dtype=np.float32)

        # Create spatial activation map from person detections
        activation = np.zeros((target_h, target_w), dtype=np.float32)
        for i in np.where(mask)[0]:
            cx, cy, w, h = boxes[i]
            # Normalize to image dimensions (YOLO outputs are normalized)
            x0 = max(0, int((cx - w / 2) * target_w))
            y0 = max(0, int((cy - h / 2) * target_h))
            x1 = min(target_w, int((cx + w / 2) * target_w))
            y1 = min(target_h, int((cy + h / 2) * target_h))
            if x1 > x0 and y1 > y0:
                activation[y0:y1, x0:x1] = max(
                    activation[y0:y1, x0:x1].max(), person_scores[i]
                )

        return activation
