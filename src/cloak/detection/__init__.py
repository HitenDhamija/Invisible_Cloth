"""Blue-cloth detection subsystem."""

from cloak.detection.adaptive import AdaptivePreprocessor
from cloak.detection.auto_calibrator import AutoCalibrator, CalibrationResult, CalibrationState
from cloak.detection.calibrator import HSVCalibrator
from cloak.detection.detector import BlueColorDetector, DetectionStats
from cloak.detection.model_manager import ModelManager, ModelManagerError
from cloak.detection.person import PersonDetector, PersonDetectorError
from cloak.detection.person_aware import PersonAwareDetector
from cloak.detection.profile_manager import ProfileManager, ProfileManagerError
from cloak.detection.segmenter import AIHybridDetector

__all__ = [
    "AdaptivePreprocessor",
    "AIHybridDetector",
    "AutoCalibrator",
    "BlueColorDetector",
    "CalibrationResult",
    "CalibrationState",
    "DetectionStats",
    "HSVCalibrator",
    "ModelManager",
    "ModelManagerError",
    "PersonAwareDetector",
    "PersonDetector",
    "PersonDetectorError",
    "ProfileManager",
    "ProfileManagerError",
]
