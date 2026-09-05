"""Calibration profile manager for saving and loading HSV ranges.

Stores calibration results as YAML files in ``configs/profiles/``.
Each profile contains the computed HSV ranges, calibration metadata,
and relevant parameters for reproducibility.

Profile format::

    name: bright_blue
    hsv_lower: [85, 100, 100]
    hsv_upper: [135, 255, 255]
    method: percentile+histogram
    pixel_count: 15000
    median_hsv: [110, 180, 200]
    iqr_hsv: [15, 40, 50]
    calibrated_at: "2025-01-15 14:30:00"
    camera_resolution: [640, 480]
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from cloak.detection.auto_calibrator import CalibrationResult

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "profiles"


class ProfileManagerError(Exception):
    """Raised when profile save/load fails."""


class ProfileManager:
    """Save, load, and list calibration profiles.

    Example::

        pm = ProfileManager()
        pm.save(result, "bright_blue")
        loaded = pm.load("bright_blue")
    """

    def __init__(self, profiles_dir: Path | str | None = None) -> None:
        self._dir = Path(profiles_dir) if profiles_dir else _PROFILES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- public API -----------------------------------------------------------

    @property
    def profiles_dir(self) -> Path:
        return self._dir

    def save(
        self,
        result: CalibrationResult,
        name: str,
        camera_resolution: tuple[int, int] | None = None,
    ) -> Path:
        """Save a calibration profile to YAML.

        Args:
            result: The calibration result to save.
            name: Profile name (used as filename).
            camera_resolution: Optional (width, height) for reference.

        Returns:
            Path to the saved profile file.
        """
        name = self._sanitize_name(name)
        path = self._dir / f"{name}.yaml"

        profile = {
            "name": name,
            "hsv_lower": result.hsv_lower,
            "hsv_upper": result.hsv_upper,
            "method": result.method,
            "pixel_count": result.pixel_count,
            "median_hsv": result.median_hsv,
            "iqr_hsv": result.iqr_hsv,
            "calibrated_at": result.timestamp,
        }

        if camera_resolution is not None:
            profile["camera_resolution"] = list(camera_resolution)

        try:
            with open(path, "w") as f:
                yaml.dump(profile, f, default_flow_style=False, sort_keys=False)
            logger.info("Profile saved: %s", path)
            return path
        except Exception as exc:
            raise ProfileManagerError(f"Failed to save profile: {exc}") from exc

    def load(self, name: str) -> dict:
        """Load a calibration profile from YAML.

        Args:
            name: Profile name (without .yaml extension).

        Returns:
            Dictionary with profile data.

        Raises:
            ProfileManagerError: If profile not found or invalid.
        """
        name = self._sanitize_name(name)
        path = self._dir / f"{name}.yaml"

        if not path.exists():
            raise ProfileManagerError(f"Profile not found: {path}")

        try:
            with open(path) as f:
                profile = yaml.safe_load(f)

            # Validate required fields
            required = ["hsv_lower", "hsv_upper"]
            for field in required:
                if field not in profile:
                    raise ProfileManagerError(f"Profile missing required field: {field}")

            logger.info("Profile loaded: %s", path)
            return profile
        except yaml.YAMLError as exc:
            raise ProfileManagerError(f"Invalid YAML in profile: {exc}") from exc

    def delete(self, name: str) -> bool:
        """Delete a calibration profile.

        Returns:
            True if deleted, False if not found.
        """
        name = self._sanitize_name(name)
        path = self._dir / f"{name}.yaml"

        if not path.exists():
            return False

        path.unlink()
        logger.info("Profile deleted: %s", path)
        return True

    def list_profiles(self) -> list[str]:
        """List all available profile names.

        Returns:
            Sorted list of profile names (without .yaml extension).
        """
        profiles = []
        for p in self._dir.glob("*.yaml"):
            profiles.append(p.stem)
        return sorted(profiles)

    def profile_exists(self, name: str) -> bool:
        """Check if a profile exists."""
        name = self._sanitize_name(name)
        return (self._dir / f"{name}.yaml").exists()

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize profile name for filesystem safety."""
        # Remove path separators and extension
        name = name.replace("/", "_").replace("\\", "_")
        if name.endswith(".yaml"):
            name = name[:-5]
        return name.strip() or "default"
