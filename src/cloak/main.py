"""Application entry point for the Blue Invisibility Cloak.

Phase 10: Polished interactive application with state machine,
recording, screenshots, help overlay, error UX, and CLI options.
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from cloak.app_state import AppState, AppStateMachine
from cloak.camera.webcam import WebcamCapture, WebcamCaptureError
from cloak.capture.model import BackgroundModel, CaptureState
from cloak.config.loader import load_config
from cloak.config.schemas import CloakConfig
from cloak.detection.adaptive import AdaptivePreprocessor
from cloak.detection.auto_calibrator import AutoCalibrator
from cloak.detection.detector import BlueColorDetector
from cloak.detection.model_manager import ModelManagerError
from cloak.detection.person import PersonDetectorError
from cloak.detection.person_aware import PersonAwareDetector
from cloak.detection.profile_manager import ProfileManager
from cloak.detection.segmenter import AIHybridDetector
from cloak.monitoring.performance import PerformanceTracker
from cloak.processing.refiner import MaskRefiner
from cloak.processing.temporal import TemporalMaskSmoother
from cloak.recording.recorder import VideoRecorder
from cloak.recording.screenshot import ScreenCapturer
from cloak.rendering.renderer import InvisibilityRenderer, RenderError
from cloak.ui.error_display import ErrorDisplay
from cloak.ui.help_overlay import HelpOverlay
from cloak.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Key constants
_KEY_Q = ord("q")
_KEY_ESC = 27
_KEY_B = ord("b")
_KEY_D = ord("d")
_KEY_C = ord("c")
_KEY_R = ord("r")
_KEY_S = ord("s")
_KEY_P = ord("p")
_KEY_M = ord("m")
_KEY_H = ord("h")
_KEY_F = ord("f")
_KEY_V = ord("v")
_KEY_G = ord("g")
_KEY_O = ord("o")
_KEY_A = ord("a")
_KEY_X = ord("x")
_KEY_PLUS = ord("+")
_KEY_MINUS = ord("-")
_KEY_EQUAL = ord("=")
_KEY_1 = ord("1")
_KEY_2 = ord("2")
_KEY_3 = ord("3")
_KEY_4 = ord("4")
_KEY_5 = ord("5")
_KEY_6 = ord("6")
_KEY_7 = ord("7")
_KEY_8 = ord("8")

_DEBUG_VIEWS = ("normal", "hsv", "mask", "region", "compare", "person", "intersection", "hybrid")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cloak",
        description="Blue Invisibility Cloak - real-time invisibility effect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m cloak.main\n"
            "  python -m cloak.main --camera 1\n"
            "  python -m cloak.main --mode ai_hybrid\n"
            "  python -m cloak.main --config configs/bright_blue.yaml --debug\n"
            "  python -m cloak.main --profile bright_blue --no-ai\n"
        ),
    )
    parser.add_argument(
        "-c", "--config",
        type=str, default=None,
        help="Path to a YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--camera",
        type=int, default=None,
        help="Camera device index (overrides config)",
    )
    parser.add_argument(
        "--video",
        type=str, default=None,
        help="Path to a video file to use instead of a live camera",
    )
    parser.add_argument(
        "--mode",
        type=str, default=None,
        choices=["hsv", "person_aware_hsv", "ai_hybrid"],
        help="Detection mode (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug view and verbose logging",
    )
    parser.add_argument(
        "--profile",
        type=str, default=None,
        help="Load a saved calibration profile by name",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI detection (force HSV-only mode)",
    )
    parser.add_argument(
        "--resolution",
        type=str, default=None,
        help="Camera resolution as WIDTHxHEIGHT (e.g. 1280x720)",
    )
    parser.add_argument(
        "--fps",
        type=int, default=None,
        help="Camera FPS (overrides config)",
    )
    return parser.parse_args()


def _apply_cli_overrides(config: CloakConfig, args: argparse.Namespace) -> CloakConfig:
    """Apply command-line overrides to the loaded config."""
    if args.camera is not None:
        config = config.model_copy(update={"camera": config.camera.model_copy(update={"device_id": args.camera})})
    if args.video is not None:
        config = config.model_copy(update={"camera": config.camera.model_copy(update={"video_path": args.video})})
    if args.mode is not None:
        config = config.model_copy(update={"detection": config.detection.model_copy(update={"mode": args.mode})})
    if args.debug:
        config = config.model_copy(update={
            "performance": config.performance.model_copy(update={"debug_mode": True, "show_perf_overlay": True}),
        })
    if args.no_ai:
        config = config.model_copy(update={
            "detection": config.detection.model_copy(update={"mode": "hsv"}),
            "ai": config.ai.model_copy(update={"enabled": False}),
        })
    if args.resolution:
        try:
            w, h = args.resolution.lower().split("x")
            config = config.model_copy(update={
                "camera": config.camera.model_copy(update={"width": int(w), "height": int(h)}),
            })
        except (ValueError, TypeError):
            logger.warning("Invalid resolution format '%s', expected WIDTHxHEIGHT", args.resolution)
    if args.fps is not None:
        config = config.model_copy(update={
            "camera": config.camera.model_copy(update={"fps": args.fps}),
        })
    return config


_WINDOW_NAME = "Blue Invisibility Cloak"


def _show_welcome_screen() -> None:
    """Display a fullscreen welcome guide with a clickable Next button."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Layout
    cx = screen_w // 2
    margin = 80

    # Title
    cv2.putText(frame, "Blue Invisibility Cloak", (cx - 260, 80), font, 1.2, (255, 200, 50), 3)
    cv2.line(frame, (margin, 110), (screen_w - margin, 110), (80, 80, 80), 2)

    # How it works
    y = 160
    cv2.putText(frame, "HOW IT WORKS:", (margin, y), font, 0.7, (0, 255, 200), 2)
    y += 40
    how_lines = [
        "1. The camera captures your background (stand still for 3 seconds)",
        "2. Hold a BLUE cloth or fabric in front of the camera",
        "3. The blue area disappears — showing the background behind it",
        "4. You appear invisible!",
    ]
    for line in how_lines:
        cv2.putText(frame, line, (margin + 20, y), font, 0.55, (220, 220, 220), 1)
        y += 35

    # Tips
    y += 15
    cv2.putText(frame, "TIPS:", (margin, y), font, 0.7, (0, 255, 200), 2)
    y += 40
    tips = [
        "Use any blue cloth — t-shirt, scarf, paper, or fabric",
        "Press B to recapture background if lighting changes",
        "Press H anytime to see all keyboard shortcuts on screen",
    ]
    for tip in tips:
        cv2.putText(frame, tip, (margin + 20, y), font, 0.55, (220, 220, 220), 1)
        y += 35

    # Key controls
    y += 15
    cv2.putText(frame, "KEY CONTROLS:", (margin, y), font, 0.7, (0, 255, 200), 2)
    y += 40

    col1_x = margin + 20
    col2_x = cx + 40
    col1 = [
        ("ESC", "Quit"),
        ("H", "Show all shortcuts"),
        ("B", "Recapture background"),
        ("P", "Pause / resume"),
        ("V", "Toggle cloak / debug view"),
    ]
    col2 = [
        ("F", "Toggle fullscreen"),
        ("R", "Start/stop recording"),
        ("S", "Take screenshot"),
        ("G", "Toggle soft/hard blend"),
        ("+/-", "Adjust blend alpha"),
    ]

    for i, (key, desc) in enumerate(col1):
        ky = y + i * 32
        cv2.putText(frame, key, (col1_x, ky), font, 0.5, (0, 220, 255), 1)
        cv2.putText(frame, desc, (col1_x + 140, ky), font, 0.5, (200, 200, 200), 1)
    for i, (key, desc) in enumerate(col2):
        ky = y + i * 32
        cv2.putText(frame, key, (col2_x, ky), font, 0.5, (0, 220, 255), 1)
        cv2.putText(frame, desc, (col2_x + 140, ky), font, 0.5, (200, 200, 200), 1)

    # Next button
    btn_w, btn_h = 220, 55
    btn_x = cx - btn_w // 2
    btn_y = screen_h - 100
    btn_rect = (btn_x, btn_y, btn_x + btn_w, btn_y + btn_h)

    clicked = [False]

    def _on_mouse(event, x, y, flags, param):
        if (event == cv2.EVENT_LBUTTONDOWN
                and btn_x <= x <= btn_x + btn_w
                and btn_y <= y <= btn_y + btn_h):
            clicked[0] = True

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(_WINDOW_NAME, _on_mouse)

    while not clicked[0]:
        overlay = frame.copy()
        cv2.rectangle(overlay, (btn_rect[0], btn_rect[1]), (btn_rect[2], btn_rect[3]), (0, 180, 80), -1)
        cv2.rectangle(overlay, (btn_rect[0], btn_rect[1]), (btn_rect[2], btn_rect[3]), (0, 220, 100), 2)
        cv2.putText(overlay, "NEXT  >", (cx - 45, btn_y + 38), font, 0.9, (255, 255, 255), 2)
        cv2.imshow(_WINDOW_NAME, overlay)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(_WINDOW_NAME)
            raise SystemExit(0)

    cv2.destroyWindow(_WINDOW_NAME)
    cv2.waitKey(1)


def run(config: CloakConfig) -> None:
    """Main application loop with state machine."""

    app = AppStateMachine()
    error_display = ErrorDisplay()
    help_overlay = HelpOverlay()
    recorder = VideoRecorder(config.output.video_dir)
    screencapper = ScreenCapturer(config.output.screenshot_dir)

    bg_model = BackgroundModel(config.background)
    bg_model.debug_enabled = config.performance.debug_mode

    # Detection setup
    person_detector = None
    hybrid_detector = None
    if config.detection.mode == "person_aware_hsv":
        try:
            person_detector = PersonAwareDetector(
                config.detection, config.processing, config.ai,
            )
            detector = person_detector.hsv_detector
        except PersonDetectorError as exc:
            error_display.show("AI model unavailable — switched to HSV", duration=6.0)
            logger.error("Failed to initialize person detector: %s", exc)
            if config.ai.fallback_to_hsv:
                config = config.model_copy(update={
                    "detection": config.detection.model_copy(update={"mode": "hsv"}),
                })
            else:
                app.force_error(str(exc))
                _run_error_loop(config, app, error_display)
                return
    elif config.detection.mode == "ai_hybrid":
        try:
            hybrid_detector = AIHybridDetector(
                config.detection, config.processing, config.ai,
            )
            detector = hybrid_detector.hsv_detector
        except ModelManagerError as exc:
            error_display.show("AI model unavailable — switched to HSV", duration=6.0)
            logger.error("Failed to initialize AI hybrid detector: %s", exc)
            if config.ai.fallback_to_hsv:
                config = config.model_copy(update={
                    "detection": config.detection.model_copy(update={"mode": "hsv"}),
                })
            else:
                app.force_error(str(exc))
                _run_error_loop(config, app, error_display)
                return
    else:
        detector = BlueColorDetector(config.detection, config.processing)

    refiner = MaskRefiner(config.mask)
    renderer = InvisibilityRenderer(config.rendering)
    auto_calibrator = AutoCalibrator(config.calibration, config.detection)
    profile_manager = ProfileManager()
    preprocessor = AdaptivePreprocessor(config.adaptive)
    smoother = TemporalMaskSmoother(config.temporal)
    perf_tracker = PerformanceTracker()

    # UI state
    debug_view = config.detection.debug_view
    show_rendered = True
    show_perf_overlay = config.performance.show_perf_overlay
    mask_stats = None
    refinement_stats = None
    frame_timestamp = 0
    debug_frame: np.ndarray | None = None

    try:
        try:
            cam_ctx = WebcamCapture(config.camera)
            cam = cam_ctx.__enter__()
        except WebcamCaptureError as exc:
            error_display.show("No camera detected — check connection", duration=8.0)
            app.force_error(str(exc))
            _run_error_loop(config, app, error_display)
            return

        _show_welcome_screen()

        # Open camera window in fullscreen
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        app.transition(AppState.BACKGROUND_CAPTURE)
        bg_model.start_capture()
        logger.info("Camera opened, starting background capture")

        while True:
            perf_tracker.start_frame()

            # Stage 1: Capture
            perf_tracker.start("capture")
            frame = None
            for _retry in range(5):
                try:
                    frame = cam.read()
                    break
                except WebcamCaptureError:
                    time.sleep(0.03)
            if frame is None:
                error_display.show("Camera disconnected", duration=5.0)
                app.force_error("Camera disconnected after retries")
                break
            perf_tracker.stop("capture")

            # Background capture phase
            display = bg_model.process_frame(frame)

            if bg_model.has_background and app.captures_background:
                app.transition(AppState.RUNNING)
                # Show ready message briefly
                error_display.show(
                    "Background ready! Now hold blue cloth in front of camera",
                    duration=5.0,
                )

            # Detection + refinement + rendering (after background ready)
            if bg_model.has_background and not app.is_paused:
                # Stage 2: Adaptive preprocessing
                detect_frame = frame
                if preprocessor.enabled:
                    perf_tracker.start("preprocess")
                    detect_frame = preprocessor.preprocess(frame)
                    perf_tracker.stop("preprocess")

                # Stage 3: Detection
                perf_tracker.start("detect")
                if hybrid_detector is not None:
                    raw_mask, mask_stats = hybrid_detector.detect(
                        detect_frame, frame_timestamp,
                    )
                elif person_detector is not None:
                    raw_mask, mask_stats = person_detector.detect(
                        detect_frame, frame_timestamp,
                    )
                else:
                    raw_mask, mask_stats = detector.detect(detect_frame)
                perf_tracker.stop("detect")

                # Stage 4: Mask refinement
                perf_tracker.start("refine")
                mask, soft_mask, refinement_stats = refiner.refine(raw_mask)
                perf_tracker.stop("refine")

                # Stage 5: Temporal smoothing
                perf_tracker.start("temporal")
                smooth_mask = smoother.smooth(mask)
                perf_tracker.stop("temporal")

                # Stage 6: Rendering
                perf_tracker.start("render")
                if show_rendered:
                    try:
                        display = renderer.render(
                            frame, bg_model.background, smooth_mask, soft_mask
                        )
                    except (RenderError, cv2.error) as exc:
                        logger.warning("Render failed: %s", exc, exc_info=True)
                        display = frame.copy()
                else:
                    display = _apply_debug_view(
                        display, frame, raw_mask, smooth_mask,
                        detector, debug_view, person_detector,
                        hybrid_detector, config,
                    )
                perf_tracker.stop("render")

                # Save debug frame for screenshot
                if config.output.screenshot_debug:
                    debug_frame = _apply_debug_view(
                        display.copy(), frame, raw_mask, smooth_mask,
                        detector, debug_view, person_detector,
                        hybrid_detector, config,
                    )

            perf_tracker.stop_frame()
            frame_timestamp += 33

            # Recording: write frame
            if recorder.is_recording:
                recorder.write(display)

            # Overlays
            if config.performance.show_fps:
                _draw_fps(display)
            if show_perf_overlay:
                _draw_perf_overlay(display, perf_tracker, config, hybrid_detector)

            _draw_status_bar(
                display, app, bg_model, debug_view,
                mask_stats, refinement_stats,
                show_rendered, renderer, smoother, config,
                recorder,
            )

            # Error message overlay
            error_display.render(display)

            # Help overlay
            help_overlay.render(display)

            cv2.imshow(_WINDOW_NAME, display)

            # Detect window close via X button
            try:
                if cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            # Calibration window dispatch
            if app.is_calibrating:
                auto_calibrator.step(frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            # Key dispatch
            if key == _KEY_ESC:
                break

            elif key == _KEY_B:
                if recorder.is_recording:
                    recorder.stop()
                    error_display.show("Recording stopped — background recapture", duration=3.0)
                bg_model.recapture()
                smoother.reset()
                perf_tracker.reset()
                if app.is_running:
                    app.transition(AppState.BACKGROUND_CAPTURE)

            elif key == _KEY_C:
                if app.is_calibrating:
                    result = auto_calibrator.collect(frame)
                    if result is None:
                        error_display.show("Calibration failed — place cloth in ROI", duration=4.0)
                elif app.is_running:
                    auto_calibrator.start(frame.shape)
                    app.transition(AppState.CALIBRATION)

            elif key == _KEY_A:
                if app.is_calibrating:
                    try:
                        lower, upper = auto_calibrator.accept()
                        detector.set_bounds(lower, upper)
                        if config.calibration.auto_save:
                            profile_manager.save(
                                auto_calibrator.result,
                                config.calibration.profile_name,
                                (config.camera.width, config.camera.height),
                            )
                        error_display.show("Calibration applied", duration=3.0)
                    except ValueError:
                        error_display.show("No calibration result to accept", duration=3.0)
                    auto_calibrator.destroy()
                    app.transition(AppState.RUNNING)

            elif key == _KEY_X:
                if app.is_calibrating:
                    auto_calibrator.cancel()
                    auto_calibrator.destroy()
                    error_display.show("Calibration cancelled", duration=2.0)
                    app.transition(AppState.RUNNING)

            elif key == _KEY_R:
                if recorder.is_recording:
                    path = recorder.stop()
                    if path:
                        error_display.show(f"Recording saved: {path.name}", duration=4.0)
                else:
                    try:
                        path = recorder.start(
                            config.camera.width, config.camera.height,
                            config.recording.fps, config.recording.codec,
                        )
                        error_display.show(f"Recording: {path.name}", duration=3.0)
                    except Exception as exc:
                        error_display.show(f"Recording could not start: {exc}", duration=5.0)

            elif key == _KEY_S:
                try:
                    paths = screencapper.capture_pair(display, debug_frame)
                    names = ", ".join(p.name for p in paths)
                    error_display.show(f"Screenshot: {names}", duration=3.0)
                except Exception as exc:
                    error_display.show(f"Screenshot failed: {exc}", duration=4.0)

            elif key == _KEY_P:
                if app.is_running:
                    app.transition(AppState.PAUSED)
                    error_display.show("Paused", duration=2.0)
                elif app.is_paused:
                    app.transition(AppState.RUNNING)

            elif key == _KEY_M:
                if hybrid_detector is not None:
                    if config.detection.mode == "ai_hybrid":
                        config = config.model_copy(update={
                            "detection": config.detection.model_copy(update={"mode": "hsv"}),
                        })
                    else:
                        config = config.model_copy(update={
                            "detection": config.detection.model_copy(update={"mode": "ai_hybrid"}),
                        })
                elif person_detector is not None:
                    if config.detection.mode == "person_aware_hsv":
                        config = config.model_copy(update={
                            "detection": config.detection.model_copy(update={"mode": "hsv"}),
                        })
                    else:
                        config = config.model_copy(update={
                            "detection": config.detection.model_copy(update={"mode": "person_aware_hsv"}),
                        })

            elif key == _KEY_D:
                bg_model.debug_enabled = not bg_model.debug_enabled

            elif key == _KEY_H:
                help_overlay.toggle()

            elif key == _KEY_V:
                show_rendered = not show_rendered

            elif key == _KEY_G:
                renderer.use_soft_blend = not renderer.use_soft_blend

            elif key == _KEY_O:
                show_perf_overlay = not show_perf_overlay

            elif key == _KEY_F:
                _toggle_fullscreen()

            elif key in (_KEY_PLUS, _KEY_EQUAL):
                new_alpha = min(1.0, renderer.blend_alpha + 0.1)
                renderer.blend_alpha = new_alpha

            elif key == _KEY_MINUS:
                new_alpha = max(0.0, renderer.blend_alpha - 0.1)
                renderer.blend_alpha = new_alpha

            elif key == _KEY_1:
                debug_view = "normal"
            elif key == _KEY_2:
                debug_view = "hsv"
            elif key == _KEY_3:
                debug_view = "mask"
            elif key == _KEY_4:
                debug_view = "region"
            elif key == _KEY_5:
                debug_view = "compare"
            elif key == _KEY_6:
                debug_view = "person"
            elif key == _KEY_7:
                debug_view = "intersection"
            elif key == _KEY_8:
                debug_view = "hybrid"

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if recorder.is_recording:
            recorder.stop()
        if hybrid_detector is not None:
            hybrid_detector.close()
        if person_detector is not None:
            person_detector.close()
        auto_calibrator.destroy()
        cam_ctx.__exit__(None, None, None)
        cv2.destroyAllWindows()
        logger.info("Application shut down")


def _run_error_loop(
    config: CloakConfig,
    app: AppStateMachine,
    error_display: ErrorDisplay,
) -> None:
    """Minimal loop showing error state until user quits."""
    while True:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(
            frame, "Blue Invisibility Cloak", (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
        )
        if app.error_message:
            cv2.putText(
                frame, app.error_message[:50], (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )
        cv2.putText(
            frame,             "Press ESC to quit, B to retry", (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
        )
        error_display.render(frame)
        cv2.imshow(_WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (_KEY_Q, _KEY_ESC):
            break
        elif key == _KEY_B:
            app.transition(AppState.BACKGROUND_CAPTURE)
            break
    cv2.destroyAllWindows()


# -- debug view rendering -----------------------------------------------------


def _apply_debug_view(
    display: np.ndarray,
    frame: np.ndarray,
    raw_mask: np.ndarray,
    refined_mask: np.ndarray,
    detector: BlueColorDetector,
    view: str,
    person_detector: PersonAwareDetector | None,
    hybrid_detector: AIHybridDetector | None,
    config: CloakConfig,
) -> np.ndarray:
    """Return the frame modified according to the active debug view."""
    if view == "hsv":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif view == "mask":
        return cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
    elif view == "region":
        return detector.detect_blue_region(frame, refined_mask)
    elif view == "compare":
        return _build_comparison_panel(frame, raw_mask, refined_mask)
    elif view == "person":
        if hybrid_detector is not None:
            person_mask = hybrid_detector.model_manager.predict(frame)
            return cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR)
        elif person_detector is not None:
            person_mask = person_detector.person_detector.detect(
                frame, int(time.time() * 1000),
            )
            vis = (person_mask * 255).astype(np.uint8)
            return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        return display
    elif view == "intersection":
        if hybrid_detector is not None:
            return _build_hybrid_panel(frame, hybrid_detector, config)
        elif person_detector is not None:
            return _build_intersection_panel(frame, person_detector, config)
        return display
    elif view == "hybrid":
        if hybrid_detector is not None:
            return _build_hybrid_panel(frame, hybrid_detector, config)
        return display
    return display


def _build_comparison_panel(
    frame: np.ndarray,
    raw_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> np.ndarray:
    h, w = frame.shape[:2]
    half_w = w // 2
    raw_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    ref_bgr = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
    raw_small = cv2.resize(raw_bgr, (half_w, h))
    ref_small = cv2.resize(ref_bgr, (half_w, h))
    combined = np.hstack([raw_small, ref_small])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "RAW mask", (10, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "REFINED mask", (half_w + 10, 30), font, 0.7, (0, 255, 255), 2)
    cv2.line(combined, (half_w, 0), (half_w, h), (255, 255, 255), 2)
    return combined


def _build_intersection_panel(
    frame: np.ndarray,
    person_detector: PersonAwareDetector,
    config: CloakConfig,
) -> np.ndarray:
    h, w = frame.shape[:2]
    third_w = w // 3
    blue_mask, _ = person_detector.hsv_detector.detect(frame)
    person_mask = person_detector.person_detector.detect(frame, int(time.time() * 1000))
    person_binary = (person_mask >= config.ai.person_threshold).astype(np.uint8) * 255
    final_mask = cv2.bitwise_and(blue_mask, person_binary)
    blue_bgr = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    person_bgr = cv2.cvtColor(person_binary, cv2.COLOR_GRAY2BGR)
    final_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    panels = np.hstack([
        cv2.resize(blue_bgr, (third_w, h)),
        cv2.resize(person_bgr, (third_w, h)),
        cv2.resize(final_bgr, (third_w, h)),
    ])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panels, "HSV mask", (10, 30), font, 0.6, (0, 255, 255), 2)
    cv2.putText(panels, "Person mask", (third_w + 10, 30), font, 0.6, (0, 255, 255), 2)
    cv2.putText(panels, "Final mask", (2 * third_w + 10, 30), font, 0.6, (0, 255, 255), 2)
    return panels


def _build_hybrid_panel(
    frame: np.ndarray,
    hybrid_detector: AIHybridDetector,
    config: CloakConfig,
) -> np.ndarray:
    h, w = frame.shape[:2]
    third_w = w // 3
    blue_mask, _ = hybrid_detector.hsv_detector.detect(frame)
    person_mask = hybrid_detector.last_person_mask
    if person_mask is None:
        person_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        person_mask = person_mask.astype(np.uint8)
    final_mask, _ = hybrid_detector.detect(frame)
    blue_bgr = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    person_bgr = cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR)
    final_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    panels = np.hstack([
        cv2.resize(blue_bgr, (third_w, h)),
        cv2.resize(person_bgr, (third_w, h)),
        cv2.resize(final_bgr, (third_w, h)),
    ])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panels, "HSV mask", (10, 30), font, 0.6, (0, 255, 255), 2)
    cv2.putText(panels, "AI person", (third_w + 10, 30), font, 0.6, (0, 255, 255), 2)
    cv2.putText(panels, "Hybrid final", (2 * third_w + 10, 30), font, 0.6, (0, 255, 255), 2)
    return panels


# -- FPS counter ---------------------------------------------------------------

_fps_frame_count = 0
_fps_last_time = time.perf_counter()


def _draw_fps(frame: cv2.Mat) -> None:  # type: ignore[type-arg]
    global _fps_frame_count, _fps_last_time
    _fps_frame_count += 1
    now = time.perf_counter()
    elapsed = now - _fps_last_time
    if elapsed >= 1.0:
        fps = _fps_frame_count / elapsed
        _fps_frame_count = 0
        _fps_last_time = now
    else:
        fps = 0.0
    if fps > 0:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


# -- performance overlay ------------------------------------------------------


def _draw_perf_overlay(
    frame: np.ndarray,
    tracker: PerformanceTracker,
    config: CloakConfig,
    hybrid_detector: AIHybridDetector | None = None,
) -> None:
    stats = tracker.get_stats()
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {stats['fps']['avg_ms']:.1f}" if stats["fps"]["avg_ms"] > 0 else "FPS: --",
        f"Frame: {stats['total']['avg_ms']:.1f}ms",
        f"Detect: {stats['detect']['avg_ms']:.1f}ms",
        f"Refine: {stats['refine']['avg_ms']:.1f}ms",
        f"Temporal: {stats['temporal']['avg_ms']:.1f}ms",
        f"Render: {stats['render']['avg_ms']:.1f}ms",
        f"Res: {config.camera.width}x{config.camera.height}",
        f"Mode: {config.detection.mode}",
    ]
    if hybrid_detector is not None and hybrid_detector.last_ai_latency_ms > 0:
        lines.append(f"AI infer: {hybrid_detector.last_ai_latency_ms:.1f}ms")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 20
    padding = 10
    panel_w = 200
    panel_h = len(lines) * line_height + padding * 2
    x0 = w - panel_w - 10
    y0 = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(lines):
        y = y0 + padding + (i + 1) * line_height - 5
        cv2.putText(frame, line, (x0 + padding, y), font, font_scale, (0, 255, 0), thickness)


# -- status bar ---------------------------------------------------------------


def _draw_status_bar(
    frame: cv2.Mat,  # type: ignore[type-arg]
    app: AppStateMachine,
    bg_model: BackgroundModel,
    debug_view: str,
    mask_stats: object,
    refinement_stats: object,
    show_rendered: bool,
    renderer: InvisibilityRenderer,
    smoother: TemporalMaskSmoother,
    config: CloakConfig,
    recorder: VideoRecorder,
) -> None:
    h = frame.shape[0]
    parts: list[str] = []

    # State indicator
    parts.append(app.label)

    if bg_model.state == CaptureState.READY:
        parts.append("BG:OK")
    elif bg_model.state in (CaptureState.CAPTURING, CaptureState.COUNTDOWN):
        parts.append("BG:wait")

    # Recording indicator
    if recorder.is_recording:
        parts.append(f"REC:{recorder.frame_count}")

    # Render mode
    if show_rendered:
        blend = "soft" if renderer.use_soft_blend else "hard"
        parts.append(f"CLOAK:{blend}")
    else:
        parts.append(f"View:{debug_view}")

    if mask_stats is not None:
        parts.append(f"Blue:{mask_stats.cloak_ratio:.1%}")

    # Detection mode
    if config.detection.mode == "ai_hybrid":
        parts.append("AI:hybrid")
    elif config.detection.mode == "person_aware_hsv":
        parts.append("AI:person")

    parts.append("H:help ESC:quit")

    status = " | ".join(parts)

    cv2.rectangle(frame, (0, h - 28), (frame.shape[1], h), (30, 30, 30), -1)
    cv2.putText(frame, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


# -- fullscreen toggle --------------------------------------------------------


_fullscreen_state = False


def _toggle_fullscreen() -> None:
    global _fullscreen_state
    _fullscreen_state = not _fullscreen_state
    flag = cv2.WINDOW_FULLSCREEN if _fullscreen_state else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, flag)


# -- entry point ---------------------------------------------------------------


def main() -> None:
    """Entry point: parse args, load config, setup logging, run."""
    args = _parse_args()
    setup_logging()
    logger.info("Blue Invisibility Cloak starting")

    config = load_config(args.config)
    config = _apply_cli_overrides(config, args)
    logger.info("Configuration loaded (mode=%s)", config.detection.mode)

    run(config)


if __name__ == "__main__":
    main()
