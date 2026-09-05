"""Tests for utility functions."""

from __future__ import annotations

import logging
from pathlib import Path

from cloak.utils.logging import setup_logging


class TestSetupLogging:
    def test_sets_root_level(self) -> None:
        setup_logging(level=logging.WARNING)
        assert logging.getLogger().level == logging.WARNING

    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "logs" / "test.log"
        setup_logging(log_file=log_file)
        assert log_file.exists()

    def test_no_crash_without_log_file(self) -> None:
        setup_logging()
        assert logging.getLogger().level == logging.INFO
