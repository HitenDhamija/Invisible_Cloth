# Demo Recording Guide

Instructions for recording a high-quality demonstration video of the Blue
Invisibility Cloak effect.

---

## Table of Contents

1. [Preparation](#1-preparation)
2. [Recording Sequence](#2-recording-sequence)
3. [Tips for Best Results](#3-tips-for-best-results)
4. [Troubleshooting](#4-troubleshooting)

---

## 1. Preparation

### Hardware

- A webcam (built-in or external, 720p or higher recommended)
- A blue cloth (solid royal blue, no patterns, no logos)
- A stable surface to mount or place the camera
- Good lighting (see tips below)

### Software

1. Launch the application:
   ```bash
   python -m cloak.main
   ```
   Or with a custom config:
   ```bash
   python -m cloak.main --config configs/my_config.json
   ```

2. Ensure the camera feed is active (you should see the live view)

3. Start recording before the demo begins (press `R` to toggle recording)

### Pre-Demo Checklist

- [ ] Camera is stable and focused
- [ ] Lighting is even (no harsh shadows)
- [ ] Blue cloth is visible and accessible
- [ ] Background is relatively static
- [ ] No other blue objects are prominently in frame
- [ ] Recording is active

---

## 2. Recording Sequence

Follow this sequence for a compelling, complete demonstration:

### Step 1: Empty Room (3-5 seconds)

Show the empty scene that will become the "background." This establishes what
the viewer will see when the invisibility effect is active.

**What to show:** The room/space without any people or the blue cloth in frame.

### Step 2: Background Capture (5-8 seconds)

Press **B** to initiate background capture. The system will count down
(default 3 seconds) and then capture 30 frames to build the background.

**What to show:** Stay completely still and out of frame during the countdown.
After capture completes, the display should show the captured background briefly.

**Tip:** Make sure no one walks through the frame during capture.

### Step 3: Person Enters (5-8 seconds)

Walk into the frame normally. Show that the live view is working and that the
person is visible.

**What to show:** A person entering the scene naturally. This demonstrates the
"before" state.

### Step 4: Blue Cloth Introduction (3-5 seconds)

Pick up or unfold the blue cloth. Hold it up so the viewer can see it clearly.

**What to show:** The blue cloth being brought into the scene. This establishes
what object will become "invisible."

### Step 5: Invisibility Effect (10-15 seconds)

Hold the blue cloth in front of the person. The system should replace the
cloth region with the captured background, making it appear invisible.

**What to show:**
- Hold the cloth steady for 2-3 seconds (stable effect)
- Slowly move the cloth side to side (tracking)
- Move the cloth closer and farther from the camera (size variation)
- If possible, have a second person react to the "invisibility"

### Step 6: Debug Mask View (5-8 seconds)

Press **D** to cycle through debug views. Show the binary mask view to
demonstrate the detection pipeline.

**What to show:**
- The raw binary mask (white = detected cloth, black = background)
- The mask should clearly show the cloth shape
- Point out clean edges and no noise (if the pipeline is working well)

### Step 7: Mode Switching (5-8 seconds)

If AI modes are available, cycle through them:
- **M** key to cycle detection modes
- Show the difference between pure HSV and AI-hybrid detection
- Show that the effect works consistently across modes

### Step 8: Calibration (Optional, 10-15 seconds)

If the effect needs tuning, show the calibration process:
- Press **C** to open calibration trackbars
- Adjust HSV bounds interactively
- Show the mask improving in real-time

This demonstrates the interactive calibration system.

### Step 9: End of Recording (3-5 seconds)

Press **R** again to stop recording. The video is saved to `outputs/videos/`.

---

## 3. Tips for Best Results

### Lighting

- **Even, diffused lighting** works best. Avoid direct sunlight or harsh
  overhead lights that create strong shadows on the cloth.
- **Consistent lighting** between background capture and effect demonstration.
  If you capture the background under bright light and then dim the lights,
  the background replacement will be noticeable.
- **Avoid flickering lights** (fluorescent tubes at certain frequencies can
  cause banding in the video).

### Camera Angle

- **Fixed camera position.** The background is captured from a single
  perspective. Moving the camera after capture will break the effect.
- **Eye-level or slightly elevated** angle shows the effect most clearly.
- **Wide enough field of view** to capture the person and cloth without
  cropping.

### Blue Cloth

- **Solid royal blue** (Hue ~100-130 in OpenCV HSV) works best with the
  default thresholds.
- **Avoid patterns, logos, or white text** on the cloth -- these create
  detection gaps.
- **Matte fabric** is better than shiny/silky material, which reflects light
  and varies in apparent color.
- **Large enough** to cover the person's torso. Small cloths are harder to
  detect cleanly.
- **No other blue objects** should be prominently in frame (blue shirts on
  other people, blue posters, blue furniture).

### Background

- **Static background** -- avoid scenes with moving objects (fans,TV screens,
  people walking by) during capture.
- **Distinct from the cloth color** -- a blue wall will confuse the detector.
- **Well-lit** so the captured background matches the lighting during the
  effect demonstration.

### Motion

- **Slow, deliberate movements** produce the cleanest effect. Fast motion can
  cause temporal lag (EMA smoothing).
- **Hold the cloth still** for a few seconds at the beginning to let the
  temporal smoother stabilize.
- **Avoid covering/uncovering the cloth rapidly** -- persistence counters
  will create ghost trails.

---

## 4. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Effect doesn't activate | HSV thresholds don't match cloth | Run calibration (press C) and adjust bounds |
| Blue background objects also invisible | Thresholds too broad | Tighten HSV bounds, increase saturation lower bound |
| Cloth edges are visible (blue outline) | Mask too tight or no feathering | Enable soft blend, increase feather_radius |
| Ghost trails when cloth moves | Persistence too high | Reduce persistence_frames to 1-2 |
| Effect flickers on/off | Temporal smoothing too weak | Increase persistence_frames or decrease ema_alpha |
| Background doesn't match | Lighting changed after capture | Recapture background under current lighting |
| Effect is slow/laggy | AI inference overhead | Switch to HSV-only mode, or increase inference_frame_skip |

---

## 5. Keyboard Reference

| Key | Action |
|-----|--------|
| `B` | Capture background |
| `R` | Start/stop recording |
| `D` | Cycle debug views |
| `M` | Cycle detection modes |
| `C` | Toggle calibration trackbars |
| `P` | Toggle performance overlay |
| `S` | Take screenshot |
| `Q` / `ESC` | Quit |
