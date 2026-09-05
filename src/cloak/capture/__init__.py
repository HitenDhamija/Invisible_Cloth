"""Background capture and estimation subsystem."""

from cloak.capture.aggregator import aggregate_mean, aggregate_median
from cloak.capture.model import BackgroundModel

__all__ = ["BackgroundModel", "aggregate_mean", "aggregate_median"]
