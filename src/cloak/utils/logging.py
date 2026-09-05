"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure root logger with console and optional file output.

    Args:
        level: Minimum logging level.
        log_file: Optional path to a log file. When provided, logs are
            written to both the console and this file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )

    logging.getLogger("cloak").info("Logging initialized at level %s", logging.getLevelName(level))
