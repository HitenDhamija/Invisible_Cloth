"""YAML-based configuration loader with validation."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from cloak.config.schemas import CloakConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> CloakConfig:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to a YAML config file. Falls back to the
            bundled ``configs/default.yaml`` when *None*.

    Returns:
        A fully validated ``CloakConfig`` instance.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning("Config file not found at %s, using defaults", path)
        return CloakConfig()

    logger.info("Loading configuration from %s", path)

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        logger.warning("Empty config file at %s, using defaults", path)
        return CloakConfig()

    config = CloakConfig.model_validate(raw)
    logger.info("Configuration loaded successfully")
    return config
