"""Application state machine for the Blue Invisibility Cloak.

Replaces scattered boolean flags with explicit state transitions.

States::

    INITIALIZING     -- app starting, loading config
    BACKGROUND_CAPTURE -- waiting for user to move out of frame
    RUNNING          -- normal cloak effect active
    PAUSED           -- processing paused (frozen output)
    CALIBRATION      -- auto or manual HSV calibration active
    ERROR            -- unrecoverable error, showing message

Transitions::

    INITIALIZING → BACKGROUND_CAPTURE  (camera opened)
    BACKGROUND_CAPTURE → RUNNING       (background captured)
    RUNNING ↔ PAUSED                   (P key)
    RUNNING → CALIBRATION              (C key)
    CALIBRATION → RUNNING              (A accept / X cancel)
    RUNNING → BACKGROUND_CAPTURE       (B key recapture)
    any → ERROR                        (camera lost, etc.)
    ERROR → BACKGROUND_CAPTURE         (B key retry)
"""

from __future__ import annotations

import enum
import logging

logger = logging.getLogger(__name__)


class AppState(enum.Enum):
    """Application states."""

    INITIALIZING = "initializing"
    BACKGROUND_CAPTURE = "background_capture"
    RUNNING = "running"
    PAUSED = "paused"
    CALIBRATION = "calibration"
    ERROR = "error"

    @property
    def label(self) -> str:
        """Human-readable state label."""
        _labels = {
            "initializing": "Starting...",
            "background_capture": "Capture Background",
            "running": "Running",
            "paused": "Paused",
            "calibration": "Calibration",
            "error": "Error",
        }
        return _labels.get(self.value, self.value)


class AppStateMachine:
    """Manage application state transitions.

    Example::

        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        assert sm.state == AppState.RUNNING
    """

    # Valid transitions: from_state -> set of allowed to_states
    _TRANSITIONS: dict[AppState, set[AppState]] = {
        AppState.INITIALIZING: {AppState.BACKGROUND_CAPTURE, AppState.ERROR},
        AppState.BACKGROUND_CAPTURE: {
            AppState.RUNNING,
            AppState.ERROR,
        },
        AppState.RUNNING: {
            AppState.PAUSED,
            AppState.CALIBRATION,
            AppState.BACKGROUND_CAPTURE,
            AppState.ERROR,
        },
        AppState.PAUSED: {
            AppState.RUNNING,
            AppState.BACKGROUND_CAPTURE,
            AppState.ERROR,
        },
        AppState.CALIBRATION: {
            AppState.RUNNING,
            AppState.BACKGROUND_CAPTURE,
            AppState.ERROR,
        },
        AppState.ERROR: {
            AppState.INITIALIZING,
            AppState.BACKGROUND_CAPTURE,
        },
    }

    def __init__(self) -> None:
        self._state = AppState.INITIALIZING
        self._error_message: str | None = None
        self._previous_state: AppState | None = None

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def previous_state(self) -> AppState | None:
        return self._previous_state

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def label(self) -> str:
        return self._state.label

    @property
    def is_running(self) -> bool:
        return self._state == AppState.RUNNING

    @property
    def is_paused(self) -> bool:
        return self._state == AppState.PAUSED

    @property
    def is_calibrating(self) -> bool:
        return self._state == AppState.CALIBRATION

    @property
    def is_error(self) -> bool:
        return self._state == AppState.ERROR

    @property
    def captures_background(self) -> bool:
        return self._state == AppState.BACKGROUND_CAPTURE

    def transition(self, to_state: AppState, error_msg: str | None = None) -> bool:
        """Attempt a state transition.

        Args:
            to_state: Desired target state.
            error_msg: Error message (required when transitioning to ERROR).

        Returns:
            True if transition succeeded, False if invalid.
        """
        allowed = self._TRANSITIONS.get(self._state, set())
        if to_state not in allowed:
            logger.warning(
                "Invalid transition: %s -> %s (allowed: %s)",
                self._state.value,
                to_state.value,
                [s.value for s in allowed],
            )
            return False

        self._previous_state = self._state
        self._state = to_state

        if to_state == AppState.ERROR:
            self._error_message = error_msg or "Unknown error"
            logger.error("State -> ERROR: %s", self._error_message)
        else:
            self._error_message = None
            logger.debug("State: %s -> %s", self._previous_state.value, to_state.value)

        return True

    def force_error(self, message: str) -> None:
        """Force an error state from any state (for critical failures)."""
        self._previous_state = self._state
        self._state = AppState.ERROR
        self._error_message = message
        logger.error("Forced ERROR: %s", message)
