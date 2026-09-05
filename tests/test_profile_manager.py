"""Tests for calibration profile manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from cloak.detection.auto_calibrator import CalibrationResult
from cloak.detection.profile_manager import ProfileManager, ProfileManagerError

# -- helpers -------------------------------------------------------------------


def _make_result(**kwargs) -> CalibrationResult:
    defaults = dict(
        hsv_lower=[85, 100, 100],
        hsv_upper=[135, 255, 255],
        method="percentile+histogram",
        pixel_count=15000,
        median_hsv=[110.0, 180.0, 200.0],
        iqr_hsv=[15.0, 40.0, 50.0],
        timestamp="2025-01-15 14:30:00",
    )
    defaults.update(kwargs)
    return CalibrationResult(**defaults)


# -- save tests ----------------------------------------------------------------


class TestProfileSave:
    """Test profile saving."""

    def test_save_creates_file(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        path = pm.save(result, "test_profile")

        assert path.exists()
        assert path.suffix == ".yaml"

    def test_save_returns_path(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        path = pm.save(result, "test_profile")

        assert path.parent == tmp_path
        assert path.name == "test_profile.yaml"

    def test_save_with_camera_resolution(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        pm.save(result, "test", camera_resolution=(640, 480))

        loaded = pm.load("test")
        assert loaded["camera_resolution"] == [640, 480]

    def test_save_sanitizes_name(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        path = pm.save(result, "../etc/passwd")

        # Should not create files outside profiles dir
        assert path.parent == tmp_path

    def test_save_extension_removed(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        path = pm.save(result, "my_profile.yaml")

        assert path.name == "my_profile.yaml"


# -- load tests ----------------------------------------------------------------


class TestProfileLoad:
    """Test profile loading."""

    def test_load_returns_dict(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        pm.save(result, "test")

        loaded = pm.load("test")
        assert isinstance(loaded, dict)

    def test_load_contains_hsv_bounds(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        pm.save(result, "test")

        loaded = pm.load("test")
        assert loaded["hsv_lower"] == [85, 100, 100]
        assert loaded["hsv_upper"] == [135, 255, 255]

    def test_load_contains_metadata(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        pm.save(result, "test")

        loaded = pm.load("test")
        assert loaded["method"] == "percentile+histogram"
        assert loaded["pixel_count"] == 15000
        assert loaded["calibrated_at"] == "2025-01-15 14:30:00"

    def test_load_nonexistent_raises(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        with pytest.raises(ProfileManagerError, match="Profile not found"):
            pm.load("nonexistent")

    def test_load_with_extension(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        pm.save(result, "test")

        # Loading with .yaml extension should work
        loaded = pm.load("test.yaml")
        assert loaded["hsv_lower"] == [85, 100, 100]


# -- list tests ----------------------------------------------------------------


class TestProfileList:
    """Test profile listing."""

    def test_list_empty(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        assert pm.list_profiles() == []

    def test_list_profiles(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        pm.save(result, "alpha")
        pm.save(result, "beta")
        pm.save(result, "gamma")

        profiles = pm.list_profiles()
        assert profiles == ["alpha", "beta", "gamma"]

    def test_list_sorted(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        pm.save(result, "zebra")
        pm.save(result, "alpha")

        profiles = pm.list_profiles()
        assert profiles == ["alpha", "zebra"]


# -- delete tests --------------------------------------------------------------


class TestProfileDelete:
    """Test profile deletion."""

    def test_delete_existing(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        pm.save(result, "test")

        assert pm.delete("test") is True
        assert not (tmp_path / "test.yaml").exists()

    def test_delete_nonexistent(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        assert pm.delete("nonexistent") is False


# -- exists tests --------------------------------------------------------------


class TestProfileExists:
    """Test profile existence check."""

    def test_exists_after_save(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        pm.save(result, "test")

        assert pm.profile_exists("test") is True

    def test_not_exists(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        assert pm.profile_exists("nonexistent") is False


# -- edge cases ----------------------------------------------------------------


class TestProfileEdgeCases:
    """Test edge cases."""

    def test_empty_name_uses_default(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        path = pm.save(result, "")

        assert path.name == "default.yaml"

    def test_special_characters_sanitized(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result = _make_result()
        path = pm.save(result, "my/profile\\name")

        # Should not create subdirectories
        assert path.parent == tmp_path

    def test_profiles_dir_created_automatically(self, tmp_path: Path):
        profiles_dir = tmp_path / "subdir" / "profiles"
        ProfileManager(profiles_dir)

        assert profiles_dir.exists()

    def test_overwrite_existing_profile(self, tmp_path: Path):
        pm = ProfileManager(tmp_path)
        result1 = _make_result(hsv_lower=[10, 10, 10])
        result2 = _make_result(hsv_lower=[20, 20, 20])

        pm.save(result1, "test")
        pm.save(result2, "test")

        loaded = pm.load("test")
        assert loaded["hsv_lower"] == [20, 20, 20]
