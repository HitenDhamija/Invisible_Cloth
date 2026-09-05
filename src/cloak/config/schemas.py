"""Pydantic configuration schemas for the Blue Invisibility Cloak."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    """Camera hardware settings."""

    device_id: int = Field(default=0, description="Camera device index")
    width: int = Field(default=640, ge=320, le=3840, description="Frame width in pixels")
    height: int = Field(default=480, ge=240, le=2160, description="Frame height in pixels")
    fps: int = Field(default=30, ge=1, le=120, description="Frames per second")
    video_path: str = Field(
        default="",
        description="Path to a video file to use instead of a live camera. Empty = use device_id.",
    )


class BackgroundConfig(BaseModel):
    """Background capture settings."""

    capture_frames: int = Field(
        default=30, ge=1, le=120, description="Number of frames to aggregate for background"
    )
    countdown_seconds: float = Field(
        default=3.0, ge=0.0, le=10.0,
        description="Seconds of countdown before capture begins"
    )
    aggregation_method: str = Field(
        default="median",
        description="Aggregation method: 'median' (robust) or 'mean'",
    )


class DetectionConfig(BaseModel):
    """Color detection settings."""

    mode: str = Field(
        default="hsv",
        description="Detection mode: 'hsv', 'person_aware_hsv', or 'ai_hybrid'",
    )
    hsv_lower: list[int] = Field(
        default=[85, 100, 100], min_length=3, max_length=3,
        description="Lower HSV bound [H, S, V]"
    )
    hsv_upper: list[int] = Field(
        default=[135, 255, 255], min_length=3, max_length=3,
        description="Upper HSV bound [H, S, V]"
    )
    debug_view: str = Field(
        default="normal",
        description="Debug view mode: normal, hsv, mask, region, compare, person, intersection, hybrid",
    )
    calibration_mode: bool = Field(
        default=False,
        description="Launch interactive HSV calibration trackbars on startup",
    )


class ProcessingConfig(BaseModel):
    """Image processing settings for the detection pipeline."""

    blur_kernel: int = Field(default=5, ge=1, le=99, description="Gaussian blur kernel size")
    morphology_kernel: int = Field(
        default=5, ge=1, le=99, description="Morphological operation kernel size"
    )


class MaskConfig(BaseModel):
    """Mask refinement settings applied after raw HSV detection."""

    median_kernel: int = Field(
        default=5, ge=1, le=99,
        description="Median blur kernel for salt-and-pepper noise removal"
    )
    morphology_kernel: int = Field(
        default=5, ge=1, le=99,
        description="Structuring element size for open/close/dilate/erode"
    )
    open_iterations: int = Field(
        default=1, ge=0, le=10,
        description="Morphological open iterations (remove small noise)"
    )
    close_iterations: int = Field(
        default=2, ge=0, le=10,
        description="Morphological close iterations (fill holes in cloth)"
    )
    dilation_iterations: int = Field(
        default=1, ge=0, le=10,
        description="Dilation iterations (expand mask boundary)"
    )
    erosion_iterations: int = Field(
        default=1, ge=0, le=10,
        description="Erosion iterations (shrink mask boundary)"
    )
    min_region_area: int = Field(
        default=500, ge=0, le=100000,
        description="Minimum contour area in pixels to keep as cloak"
    )
    feather_radius: int = Field(
        default=7, ge=0, le=50,
        description="Gaussian blur radius for soft mask edges (0 = disabled)"
    )


class TemporalConfig(BaseModel):
    """Temporal mask smoothing settings."""

    enabled: bool = Field(
        default=True,
        description="Enable temporal mask smoothing to reduce flicker",
    )
    ema_alpha: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description=(
            "EMA weight for current frame. "
            "1.0 = no smoothing, 0.0 = frozen. "
            "Lower values increase stability but add lag."
        ),
    )
    persistence_frames: int = Field(
        default=3, ge=0, le=30,
        description=(
            "Frames a recently-active mask pixel stays ON after disappearing. "
            "0 = no persistence. Higher values reduce flicker but risk ghost trails."
        ),
    )


class RenderingConfig(BaseModel):
    """Compositing / rendering settings."""

    use_soft_blend: bool = Field(
        default=False,
        description="Use soft alpha blending instead of hard binary replacement",
    )
    blend_alpha: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Blend weight for soft mode (1.0=fully background, 0.0=fully live)",
    )


class AdaptiveConfig(BaseModel):
    """Optional illumination compensation settings."""

    enabled: bool = Field(
        default=False,
        description="Enable adaptive preprocessing to compensate for lighting changes",
    )
    clahe_clip: float = Field(
        default=2.0, ge=0.5, le=10.0,
        description="CLAHE clip limit (higher = more contrast enhancement)",
    )
    clahe_grid: int = Field(
        default=8, ge=2, le=32,
        description="CLAHE tile grid size (higher = more local adaptation)",
    )
    brightness_normalize: bool = Field(
        default=True,
        description="Normalize V-channel brightness before detection",
    )


class PerformanceConfig(BaseModel):
    """Performance and display settings."""

    show_fps: bool = Field(default=True, description="Display FPS counter on output")
    show_perf_overlay: bool = Field(
        default=False,
        description="Display per-stage timing overlay (toggle with P key at runtime)",
    )
    debug_mode: bool = Field(default=False, description="Show debug panels")


class AIConfig(BaseModel):
    """Person-aware detection settings (requires mediapipe or onnxruntime)."""

    model_config = {"protected_namespaces": ()}

    enabled: bool = Field(
        default=False,
        description="Enable person-aware detection mode",
    )
    fallback_to_hsv: bool = Field(
        default=True,
        description="Fall back to pure HSV if person detection fails",
    )
    person_threshold: float = Field(
        default=0.5, ge=0.1, le=0.9,
        description="Threshold for person segmentation mask (0.0-1.0)",
    )
    model_complexity: int = Field(
        default=0, ge=0, le=2,
        description="Pose model complexity: 0=lite, 1=full, 2=heavy",
    )
    min_detection_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence for pose detection",
    )
    min_tracking_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence for pose tracking",
    )

    # Phase 8: AI hybrid mode settings
    hybrid_model_path: str = Field(
        default="",
        description=(
            "Path to ONNX person segmentation model (e.g. yolov8n-seg.onnx). "
            "Empty string uses auto-download."
        ),
    )
    inference_width: int = Field(
        default=320, ge=160, le=1920,
        description="Inference width for AI model (lower = faster)",
    )
    inference_height: int = Field(
        default=240, ge=120, le=1080,
        description="Inference height for AI model (lower = faster)",
    )
    inference_frame_skip: int = Field(
        default=1, ge=1, le=10,
        description="Run AI inference every N frames (1 = every frame)",
    )
    use_half_precision: bool = Field(
        default=False,
        description="Use FP16 inference where supported (GPU only)",
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0.1, le=0.9,
        description="Confidence threshold for AI person segmentation",
    )


class CalibrationConfig(BaseModel):
    """Automatic HSV calibration settings."""

    roi_fraction: float = Field(
        default=0.25, ge=0.05, le=0.8,
        description="ROI size as fraction of frame (0.05-0.8)",
    )
    percentile_low: float = Field(
        default=2.0, ge=0.0, le=25.0,
        description="Lower percentile for outlier rejection (0-25)",
    )
    percentile_high: float = Field(
        default=98.0, ge=75.0, le=100.0,
        description="Upper percentile for outlier rejection (75-100)",
    )
    min_pixels: int = Field(
        default=100, ge=10, le=10000,
        description="Minimum pixel count required for valid calibration",
    )
    h_margin: int = Field(
        default=8, ge=0, le=30,
        description="Hue margin added to computed range",
    )
    s_margin: int = Field(
        default=15, ge=0, le=60,
        description="Saturation margin added to computed range",
    )
    v_margin: int = Field(
        default=15, ge=0, le=60,
        description="Value margin added to computed range",
    )
    use_kmeans: bool = Field(
        default=False,
        description="Use K-means clustering to find dominant color cluster",
    )
    kmeans_n_clusters: int = Field(
        default=3, ge=2, le=8,
        description="Number of clusters for K-means",
    )
    use_histogram: bool = Field(
        default=True,
        description="Use histogram peak analysis for range refinement",
    )
    histogram_bins: int = Field(
        default=50, ge=10, le=200,
        description="Number of histogram bins per channel",
    )
    auto_save: bool = Field(
        default=True,
        description="Automatically save calibration profile on accept",
    )
    profile_name: str = Field(
        default="default",
        description="Profile name for save/load",
    )


class RecordingConfig(BaseModel):
    """Video recording settings."""

    codec: str = Field(
        default="mp4v",
        description="FourCC codec for video recording (mp4v, XVID, MJPG)",
    )
    fps: float = Field(
        default=30.0, ge=1.0, le=120.0,
        description="Recording frame rate",
    )
    record_debug: bool = Field(
        default=False,
        description="Also record the debug view alongside the render",
    )


class OutputConfig(BaseModel):
    """Output file paths and screenshot settings."""

    video_dir: str = Field(
        default="outputs/videos",
        description="Directory for video recordings",
    )
    screenshot_dir: str = Field(
        default="outputs/screenshots",
        description="Directory for screenshots",
    )
    screenshot_quality: int = Field(
        default=95, ge=1, le=100,
        description="JPEG quality for screenshots (1-100)",
    )
    screenshot_debug: bool = Field(
        default=False,
        description="Also save a debug view screenshot when taking screenshots",
    )


class CloakConfig(BaseModel):
    """Root configuration for the application."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
