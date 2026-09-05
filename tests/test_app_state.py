"""Tests for the application state machine."""

from __future__ import annotations

import pytest

from cloak.app_state import AppState, AppStateMachine

# -- initial state -------------------------------------------------------------


class TestInitialState:
    def test_starts_in_initializing(self) -> None:
        sm = AppStateMachine()
        assert sm.state == AppState.INITIALIZING

    def test_previous_state_is_none(self) -> None:
        sm = AppStateMachine()
        assert sm.previous_state is None

    def test_error_message_is_none(self) -> None:
        sm = AppStateMachine()
        assert sm.error_message is None


# -- valid transitions ---------------------------------------------------------


class TestValidTransitions:
    def test_initializing_to_background_capture(self) -> None:
        sm = AppStateMachine()
        assert sm.transition(AppState.BACKGROUND_CAPTURE) is True
        assert sm.state == AppState.BACKGROUND_CAPTURE

    def test_background_capture_to_running(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.transition(AppState.RUNNING) is True
        assert sm.state == AppState.RUNNING

    def test_running_to_paused(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        assert sm.transition(AppState.PAUSED) is True
        assert sm.state == AppState.PAUSED

    def test_paused_to_running(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.PAUSED)
        assert sm.transition(AppState.RUNNING) is True
        assert sm.state == AppState.RUNNING

    def test_running_to_calibration(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        assert sm.transition(AppState.CALIBRATION) is True
        assert sm.state == AppState.CALIBRATION

    def test_calibration_to_running(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.CALIBRATION)
        assert sm.transition(AppState.RUNNING) is True
        assert sm.state == AppState.RUNNING

    def test_running_to_background_capture(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        assert sm.transition(AppState.BACKGROUND_CAPTURE) is True
        assert sm.state == AppState.BACKGROUND_CAPTURE

    def test_error_to_background_capture(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.ERROR, error_msg="fail")
        assert sm.transition(AppState.BACKGROUND_CAPTURE) is True
        assert sm.state == AppState.BACKGROUND_CAPTURE

    def test_error_to_initializing(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.ERROR, error_msg="fail")
        assert sm.transition(AppState.INITIALIZING) is True
        assert sm.state == AppState.INITIALIZING

    @pytest.mark.parametrize(
        "from_state",
        [
            AppState.INITIALIZING,
            AppState.BACKGROUND_CAPTURE,
            AppState.RUNNING,
            AppState.PAUSED,
            AppState.CALIBRATION,
        ],
    )
    def test_any_to_error(self, from_state: AppState) -> None:
        sm = AppStateMachine()
        if from_state != AppState.INITIALIZING:
            sm.transition(AppState.BACKGROUND_CAPTURE)
        if from_state not in (AppState.INITIALIZING, AppState.BACKGROUND_CAPTURE):
            sm.transition(AppState.RUNNING)
        if from_state == AppState.PAUSED:
            sm.transition(AppState.PAUSED)
        if from_state == AppState.CALIBRATION:
            sm.transition(AppState.CALIBRATION)
        assert sm.state == from_state
        assert sm.transition(AppState.ERROR, error_msg="boom") is True
        assert sm.state == AppState.ERROR


# -- invalid transitions -------------------------------------------------------


class TestInvalidTransitions:
    def test_initializing_to_running(self) -> None:
        sm = AppStateMachine()
        assert sm.transition(AppState.RUNNING) is False
        assert sm.state == AppState.INITIALIZING

    def test_initializing_to_paused(self) -> None:
        sm = AppStateMachine()
        assert sm.transition(AppState.PAUSED) is False

    def test_initializing_to_calibration(self) -> None:
        sm = AppStateMachine()
        assert sm.transition(AppState.CALIBRATION) is False

    def test_background_capture_to_paused(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.transition(AppState.PAUSED) is False
        assert sm.state == AppState.BACKGROUND_CAPTURE

    def test_background_capture_to_calibration(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.transition(AppState.CALIBRATION) is False

    def test_paused_to_calibration(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.PAUSED)
        assert sm.transition(AppState.CALIBRATION) is False

    def test_paused_to_error(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.PAUSED)
        assert sm.transition(AppState.CALIBRATION) is False

    def test_error_to_running(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.ERROR, error_msg="fail")
        assert sm.transition(AppState.RUNNING) is False

    def test_error_to_paused(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.ERROR, error_msg="fail")
        assert sm.transition(AppState.PAUSED) is False

    def test_invalid_transition_preserves_state(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.CALIBRATION)
        sm.transition(AppState.PAUSED)  # invalid from CALIBRATION
        assert sm.state == AppState.CALIBRATION


# -- error message handling ----------------------------------------------------


class TestErrorMessage:
    def test_error_message_set_on_error_transition(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.ERROR, error_msg="camera lost")
        assert sm.error_message == "camera lost"

    def test_default_error_message_when_none_provided(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.ERROR)
        assert sm.error_message == "Unknown error"

    def test_error_message_cleared_when_leaving_error(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.ERROR, error_msg="fail")
        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.error_message is None

    def test_error_message_cleared_on_any_non_error_transition(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.ERROR, error_msg="fail")
        sm.transition(AppState.INITIALIZING)
        assert sm.error_message is None


# -- force_error ---------------------------------------------------------------


class TestForceError:
    def test_force_error_from_initializing(self) -> None:
        sm = AppStateMachine()
        sm.force_error("critical")
        assert sm.state == AppState.ERROR
        assert sm.error_message == "critical"

    def test_force_error_from_background_capture(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.force_error("camera died")
        assert sm.state == AppState.ERROR
        assert sm.error_message == "camera died"

    def test_force_error_from_running(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.force_error("oom")
        assert sm.state == AppState.ERROR
        assert sm.error_message == "oom"

    def test_force_error_from_paused(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.PAUSED)
        sm.force_error("fatal")
        assert sm.state == AppState.ERROR
        assert sm.error_message == "fatal"

    def test_force_error_from_calibration(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.CALIBRATION)
        sm.force_error("calibration broke")
        assert sm.state == AppState.ERROR
        assert sm.error_message == "calibration broke"

    def test_force_error_from_error(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.force_error("first")
        sm.force_error("second")
        assert sm.state == AppState.ERROR
        assert sm.error_message == "second"

    def test_force_error_tracks_previous_state(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.force_error("boom")
        assert sm.previous_state == AppState.RUNNING


# -- previous_state tracking ---------------------------------------------------


class TestPreviousState:
    def test_previous_state_after_single_transition(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.previous_state == AppState.INITIALIZING

    def test_previous_state_after_multiple_transitions(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.PAUSED)
        assert sm.previous_state == AppState.RUNNING

    def test_previous_state_after_successful_error_transition(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.ERROR, error_msg="fail")
        assert sm.previous_state == AppState.RUNNING

    def test_previous_state_unchanged_on_invalid_transition(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.CALIBRATION)
        sm.transition(AppState.PAUSED)  # invalid
        assert sm.previous_state == AppState.RUNNING


# -- convenience properties ----------------------------------------------------


class TestConvenienceProperties:
    def test_is_running(self) -> None:
        sm = AppStateMachine()
        assert sm.is_running is False
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        assert sm.is_running is True

    def test_is_paused(self) -> None:
        sm = AppStateMachine()
        assert sm.is_paused is False
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.PAUSED)
        assert sm.is_paused is True

    def test_is_calibrating(self) -> None:
        sm = AppStateMachine()
        assert sm.is_calibrating is False
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.CALIBRATION)
        assert sm.is_calibrating is True

    def test_is_error(self) -> None:
        sm = AppStateMachine()
        assert sm.is_error is False
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.ERROR, error_msg="fail")
        assert sm.is_error is True

    def test_captures_background(self) -> None:
        sm = AppStateMachine()
        assert sm.captures_background is False
        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.captures_background is True

    def test_all_false_in_initializing(self) -> None:
        sm = AppStateMachine()
        assert sm.is_running is False
        assert sm.is_paused is False
        assert sm.is_calibrating is False
        assert sm.is_error is False
        assert sm.captures_background is False


# -- label property ------------------------------------------------------------


class TestLabel:
    def test_machine_label_matches_state_label(self) -> None:
        sm = AppStateMachine()
        assert sm.label == sm.state.label

    def test_label_updates_with_transitions(self) -> None:
        sm = AppStateMachine()
        assert sm.label == "Starting..."
        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.label == "Capture Background"
        sm.transition(AppState.RUNNING)
        assert sm.label == "Running"
        sm.transition(AppState.PAUSED)
        assert sm.label == "Paused"
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.CALIBRATION)
        assert sm.label == "Calibration"
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.ERROR, error_msg="fail")
        assert sm.label == "Error"


# -- state enum label ----------------------------------------------------------


class TestStateEnumLabel:
    def test_initializing_label(self) -> None:
        assert AppState.INITIALIZING.label == "Starting..."

    def test_background_capture_label(self) -> None:
        assert AppState.BACKGROUND_CAPTURE.label == "Capture Background"

    def test_running_label(self) -> None:
        assert AppState.RUNNING.label == "Running"

    def test_paused_label(self) -> None:
        assert AppState.PAUSED.label == "Paused"

    def test_calibration_label(self) -> None:
        assert AppState.CALIBRATION.label == "Calibration"

    def test_error_label(self) -> None:
        assert AppState.ERROR.label == "Error"

    def test_all_labels_are_strings(self) -> None:
        for state in AppState:
            assert isinstance(state.label, str)


# -- full lifecycle ------------------------------------------------------------


class TestFullLifecycle:
    def test_normal_flow(self) -> None:
        sm = AppStateMachine()
        assert sm.state == AppState.INITIALIZING

        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.state == AppState.BACKGROUND_CAPTURE
        assert sm.captures_background is True

        sm.transition(AppState.RUNNING)
        assert sm.state == AppState.RUNNING
        assert sm.is_running is True

        sm.transition(AppState.PAUSED)
        assert sm.state == AppState.PAUSED
        assert sm.is_paused is True

        sm.transition(AppState.RUNNING)
        assert sm.state == AppState.RUNNING

        sm.transition(AppState.CALIBRATION)
        assert sm.state == AppState.CALIBRATION
        assert sm.is_calibrating is True

        sm.transition(AppState.RUNNING)
        assert sm.state == AppState.RUNNING

    def test_error_and_recovery(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.ERROR, error_msg="camera lost")
        assert sm.is_error is True
        assert sm.error_message == "camera lost"

        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.is_error is False
        assert sm.error_message is None
        assert sm.captures_background is True

    def test_recapture_flow(self) -> None:
        sm = AppStateMachine()
        sm.transition(AppState.BACKGROUND_CAPTURE)
        sm.transition(AppState.RUNNING)
        sm.transition(AppState.BACKGROUND_CAPTURE)
        assert sm.captures_background is True
        sm.transition(AppState.RUNNING)
        assert sm.is_running is True
