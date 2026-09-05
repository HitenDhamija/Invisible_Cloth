# Algorithm Reference

Technical deep-dive into every algorithm used in the Blue Invisibility Cloak
pipeline, from color-space segmentation through compositing.

---

## Table of Contents

1. [HSV Color-Space Segmentation](#1-hsv-color-space-segmentation)
2. [Morphological Processing](#2-morphological-processing)
3. [Background Capture and Replacement](#3-background-capture-and-replacement)
4. [Alpha Compositing](#4-alpha-compositing)
5. [Temporal Smoothing](#5-temporal-smoothing)

---

## 1. HSV Color-Space Segmentation

### Why HSV over BGR/RGB

In BGR/RGB a single color is spread across three channels. A bright blue pixel
`(200, 50, 30)` and a shadowed blue pixel `(100, 25, 15)` are far apart in
Euclidean distance even though both are perceptually "blue." Lighting changes
shift all three channels simultaneously, making fixed-threshold segmentation
fragile.

HSV decouples **chrominance** (Hue) from **intensity** (Value) and **purity**
(Saturation). A blue pixel always has Hue in the range 85-135 regardless of
brightness, so a single Hue range covers bright blue, dark blue, and shadowed
folds.

### Conversion

OpenCV reads frames in BGR order. The conversion to HSV is:

$$
\begin{aligned}
H &= \text{hue angle} \in [0, 180] \quad (\text{OpenCV scale}) \\
S &= \text{saturation} \in [0, 255] \\
V &= \text{value (brightness)} \in [0, 255]
\end{aligned}
$$

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

### Thresholding

A pixel $p$ is classified as "blue cloth" if and only if:

$$
p \in \text{mask} \iff
\begin{cases}
H_{\text{lower}} \leq H(p) \leq H_{\text{upper}} \\
S_{\text{lower}} \leq S(p) \leq S_{\text{upper}} \\
V_{\text{lower}} \leq V(p) \leq V_{\text{upper}}
\end{cases}
$$

Implemented with `cv2.inRange(hsv, lower, upper)`, which returns a binary mask
(uint8, values 0 or 255).

### Default Bounds

| Channel | Lower | Upper | Rationale |
|---------|-------|-------|-----------|
| H | 85 | 135 | Covers cyan-to-purple blue range |
| S | 100 | 255 | Excludes grey/desaturated noise |
| V | 100 | 255 | Excludes very dark pixels |

### Safety Check

If more than 85% of pixels are detected as blue, the thresholds are likely too
broad. A `DetectionStats.warning` is emitted:

$$
\text{cloak\_ratio} = \frac{\text{count}(\text{mask} > 0)}{H \times W}
$$

If $\text{cloak\_ratio} > 0.85$, a warning is logged.

---

## 2. Morphological Processing

Morphological operations use a **structuring element** (kernel) to reshape binary
masks. All operations in this project use elliptical kernels for smooth, natural
boundaries.

### 2.1 Pre-Detection Smoothing (Gaussian Blur)

Before HSV conversion, an optional Gaussian blur reduces high-frequency noise:

$$
G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
$$

$$
\text{smoothed} = G * \text{frame}
$$

Kernel size is configurable (`blur_kernel`). Odd values only; even values are
incremented by 1.

### 2.2 Morphological Opening (Erosion then Dilation)

**Purpose:** Remove small isolated noise blobs (tiny blue objects) while
preserving large regions.

$$
A \circ B = (A \ominus B) \oplus B
$$

Where:
- $A$ = binary mask
- $B$ = structuring element
- $\ominus$ = erosion
- $\oplus$ = dilation

**Effect:** Erosion kills small foreground regions; dilation restores the size
of surviving regions. Net result: small noise is eliminated, large regions are
preserved.

### 2.3 Morphological Closing (Dilation then Erosion)

**Purpose:** Fill small holes inside the detected cloth (folds, wrinkles where
saturation briefly drops).

$$
A \bullet B = (A \oplus B) \ominus B
$$

**Effect:** Dilation fills gaps; erosion restores the original boundary size.
Net result: internal holes are closed without changing region extent.

### 2.4 Dilation

**Purpose:** Expand the mask boundary outward to ensure full cloth coverage and
eliminate thin un-masked borders.

$$
(A \oplus B)(x) = \max_{b \in B} A(x - b)
$$

### 2.5 Erosion

**Purpose:** Shrink the mask boundary back, smoothing edges. When dilation and
erosion use the same kernel and iteration count, the net effect is boundary
smoothing. If dilation exceeds erosion, the mask stays slightly expanded for
better coverage.

$$
(A \ominus B)(x) = \min_{b \in B} A(x + b)
$$

### 2.6 Median Blur (Salt-and-Pepper Removal)

Applied before morphological operations in the refinement pipeline. Replaces
each pixel with the median of its neighborhood, effectively removing isolated
flickering pixels without blurring edges as aggressively as Gaussian blur.

$$
\text{output}(x, y) = \text{median}\{p_i \mid p_i \in N_k(x, y)\}
$$

### 2.7 Contour Filtering (Connected Component Analysis)

After morphological cleanup, connected components smaller than
`min_region_area` are rejected:

```python
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
for label_id in range(1, num_labels):
    area = stats[label_id, cv2.CC_STAT_AREA]
    if area >= min_area:
        result[labels == label_id] = 255
```

This prevents small blue objects (a pen, a book spine) from becoming invisible.

### Refinement Pipeline Order

The full refinement pipeline applies these steps in sequence:

1. **Median blur** -- salt-and-pepper removal
2. **Morphological OPEN** -- remove small noise blobs
3. **Morphological CLOSE** -- fill small holes
4. **Dilation** -- expand boundary for coverage
5. **Erosion** -- smooth boundary
6. **Contour filtering** -- reject small components
7. **Soft mask generation** -- Gaussian-feathered edges (optional)

---

## 3. Background Capture and Replacement

### Background Capture

When the user initiates background capture, the system aggregates multiple
frames to produce a clean background image:

**Median aggregation** (default, robust to moving objects):

$$
B(x, y, c) = \text{median}\{F_i(x, y, c) \mid i = 1, \ldots, N\}
$$

Where $N$ = `capture_frames` (default 30) and $c$ is the color channel.

**Mean aggregation** (alternative):

$$
B(x, y, c) = \frac{1}{N} \sum_{i=1}^{N} F_i(x, y, c)
$$

A countdown (`countdown_seconds`, default 3.0s) gives the user time to leave
the frame before capture begins.

### Background Replacement

The background is stored as a static reference frame. During rendering, pixels
classified as "cloak" are replaced with the corresponding background pixels.

---

## 4. Alpha Compositing

Two compositing modes are available: hard (bitwise) and soft (alpha blend).

### 4.1 Hard Composite (Bitwise)

Pixel-exact replacement using boolean operations:

$$
\begin{aligned}
M_{\text{inv}} &= \neg M \\
R_{\text{cloak}} &= B \land M \\
R_{\text{visible}} &= F \land M_{\text{inv}} \\
\text{output} &= R_{\text{cloak}} \lor R_{\text{visible}}
\end{aligned}
$$

Where:
- $M$ = binary cloak mask (0 or 255)
- $B$ = background frame
- $F$ = live frame

**Effect:** Every cloak pixel is replaced 1:1 with the corresponding
background pixel. Produces crisp, pixel-exact boundaries.

### 4.2 Soft Blend (Alpha)

Smooth transition at mask boundaries using alpha compositing:

$$
\text{output}(x, y, c) = \alpha(x, y) \cdot B(x, y, c) + (1 - \alpha(x, y)) \cdot F(x, y, c)
$$

Where $\alpha(x, y) \in [0.0, 1.0]$ is the soft mask value at pixel $(x, y)$.

- $\alpha = 1.0$: fully background (complete invisibility)
- $\alpha = 0.0$: fully live frame (no effect)
- $0 < \alpha < 1$: partial transparency (smooth edge transition)

The soft mask is generated by Gaussian feathering of the binary mask:

$$
\alpha = \text{GaussianBlur}(M / 255.0, \sigma = r)
$$

Where $r$ = `feather_radius` (default 7). This produces a smooth gradient at
cloth boundaries, hiding imperfect mask edges and eliminating visible blue
outlines.

---

## 5. Temporal Smoothing

Frame-to-frame flicker is reduced using two complementary techniques: Exponential
Moving Average (EMA) and per-pixel persistence.

### 5.1 Exponential Moving Average (EMA)

The accumulated mask is updated each frame:

$$
A_t = \alpha \cdot M_t + (1 - \alpha) \cdot A_{t-1}
$$

Where:
- $A_t$ = accumulated float mask at frame $t$
- $M_t$ = current binary mask (0.0 or 1.0)
- $\alpha$ = `ema_alpha` (default 0.6)

Thresholding converts back to binary:

$$
\text{smoothed}(x, y) =
\begin{cases}
255 & \text{if } A_t(x, y) \geq \alpha \cdot 127.0 \\
0 & \text{otherwise}
\end{cases}
$$

**Parameter interpretation:**
- $\alpha = 1.0$: no smoothing (current frame only)
- $\alpha = 0.0$: frozen (never updates)
- $\alpha = 0.6$: moderate smoothing (default)

Lower $\alpha$ increases stability but adds latency (lag behind real motion).

### 5.2 Per-Pixel Persistence

Persistence counters prevent rapid disappearance of cloak pixels that fail
detection for 1-2 frames due to noise or occlusion.

For each pixel:

$$
P_t(x, y) =
\begin{cases}
N & \text{if } M_t(x, y) = 255 \text{ (active pixel resets counter)} \\
P_{t-1}(x, y) - 1 & \text{if } M_t(x, y) = 0 \text{ and } P_{t-1}(x, y) > 0 \\
0 & \text{otherwise}
\end{cases}
$$

Where $N$ = `persistence_frames` (default 3).

Pixels with $P_t > 0$ are forced ON regardless of the current detection:

$$
\text{output}(x, y) = 255 \quad \text{if } P_t(x, y) > 0
$$

**Effect:** A cloak pixel that disappears for 1-2 frames stays visible,
eliminating flickering edges. Higher values reduce flicker but risk "ghost
trails" when the cloth moves away.

### Combined Pipeline

Each frame:

1. Run EMA accumulation on the raw binary mask
2. Threshold to binary
3. Apply persistence counters (force recently-active pixels ON)
4. Return smoothed binary mask

The smoother is reset (all state cleared) when the user recaptures the background.

---

## Summary: Full Pipeline

```
Input Frame (BGR)
    |
    v
[1] Gaussian Blur (optional noise reduction)
    |
    v
[2] BGR -> HSV Conversion
    |
    v
[3] HSV Thresholding (cv2.inRange)
    |
    v
[4] Morphological Open + Close (noise removal, hole filling)
    |
    v
[5] Mask Refinement (median blur, dilation, erosion, contour filtering)
    |
    v
[6] Soft Mask Generation (Gaussian feathering)
    |
    v
[7] Temporal Smoothing (EMA + persistence)
    |
    v
[8] Alpha Compositing (hard or soft blend with background)
    |
    v
Output Frame (BGR)
```
