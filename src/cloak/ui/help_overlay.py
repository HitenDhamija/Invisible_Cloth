"""In-application help overlay displaying keyboard controls.

Togglable panel that shows all available controls overlaid on the frame.

Usage::

    help_overlay = HelpOverlay()
    help_overlay.toggle()
    help_overlay.render(frame)
"""

from __future__ import annotations

import cv2
import numpy as np

# Control definitions: (key, description)
_CONTROLS = [
    ("ESC", "Quit"),
    ("B", "Recapture background"),
    ("C", "Calibrate cloak"),
    ("A", "Accept calibration"),
    ("X", "Cancel calibration"),
    ("M", "Switch detection mode"),
    ("R", "Start/stop recording"),
    ("S", "Take screenshot"),
    ("P", "Pause / resume"),
    ("H", "Toggle this help"),
    ("V", "Toggle rendered / debug view"),
    ("O", "Toggle performance overlay"),
    ("G", "Toggle soft/hard blend"),
    ("D", "Toggle background debug"),
    ("F", "Toggle fullscreen"),
    ("+", "Increase blend alpha"),
    ("-", "Decrease blend alpha"),
    ("1-8", "Select debug view (when V is off)"),
]


class HelpOverlay:
    """Toggleable help panel showing keyboard controls.

    Example::

        overlay = HelpOverlay()
        while True:
            if key == _KEY_H:
                overlay.toggle()
            overlay.render(display)
    """

    def __init__(self) -> None:
        self._visible = False

    @property
    def visible(self) -> bool:
        return self._visible

    def toggle(self) -> None:
        """Toggle visibility."""
        self._visible = not self._visible

    def show(self) -> None:
        self._visible = True

    def hide(self) -> None:
        self._visible = False

    def render(self, frame: np.ndarray) -> None:
        """Draw the help overlay on the frame (in-place).

        Args:
            frame: The frame to draw on (modified in-place).
        """
        if not self._visible:
            return

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Panel dimensions
        line_height = 26
        padding = 16
        title_height = 36
        panel_h = title_height + len(_CONTROLS) * line_height + padding * 2
        panel_w = 320

        # Center the panel
        x0 = (w - panel_w) // 2
        y0 = (h - panel_h) // 2

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)

        # Title
        title = "Keyboard Controls"
        title_size = cv2.getTextSize(title, font, 0.65, 2)[0]
        title_x = x0 + (panel_w - title_size[0]) // 2
        cv2.putText(
            frame, title, (title_x, y0 + title_height - 8),
            font, 0.65, (0, 255, 255), 2,
        )

        # Divider
        cv2.line(
            frame,
            (x0 + padding, y0 + title_height),
            (x0 + panel_w - padding, y0 + title_height),
            (60, 60, 60), 1,
        )

        # Controls
        for i, (key, desc) in enumerate(_CONTROLS):
            y = y0 + title_height + padding + (i + 1) * line_height - 4

            # Key in yellow
            cv2.putText(
                frame, key, (x0 + padding, y),
                font, 0.48, (0, 220, 255), 1,
            )

            # Description in white
            key_width = cv2.getTextSize(key, font, 0.48, 1)[0][0]
            cv2.putText(
                frame, desc, (x0 + padding + key_width + 16, y),
                font, 0.48, (220, 220, 220), 1,
            )

        # Footer
        footer = "Press H to close"
        footer_size = cv2.getTextSize(footer, font, 0.4, 1)[0]
        footer_x = x0 + (panel_w - footer_size[0]) // 2
        cv2.putText(
            frame, footer, (footer_x, y0 + panel_h - 10),
            font, 0.4, (120, 120, 120), 1,
        )
