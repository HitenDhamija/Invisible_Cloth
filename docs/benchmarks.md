# Performance Benchmarks

How to run benchmarks, what metrics are measured, and how to interpret results.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [What Is Measured](#2-what-is-measured)
3. [Per-Stage Breakdown](#3-per-stage-breakdown)
4. [Running Benchmarks](#4-running-benchmarks)
5. [Current Results](#5-current-results)

---

## 1. Quick Start

Run the default benchmark (150 frames at 640x480):

```bash
python benchmarks/benchmark_pipeline.py
```

This generates synthetic frames and measures the full pipeline without requiring
a webcam or GPU.

---

## 2. What Is Measured

### Frames Per Second (FPS)

The primary throughput metric. Higher is better.

$$
\text{FPS} = \frac{1000}{\text{total\_latency\_ms}}
$$

Reported as:
- **Mean FPS:** Average across all measured frames
- **Median FPS:** 50th percentile (typical performance)
- **P95 FPS:** 95th percentile (worst 5% of frames)

### Frame Latency

Per-frame processing time in milliseconds. Lower is better.

$$
\text{latency} = t_{\text{end}} - t_{\text{start}}
$$

Reported as:
- **Mean:** Average latency
- **Median:** 50th percentile
- **P95:** 95th percentile (worst-case for real-time applications)

### Per-Stage Timing

Each frame is broken down into five processing stages:

| Stage | Description | What It Measures |
|-------|-------------|------------------|
| `preprocess` | BGR to HSV conversion | Color-space conversion overhead |
| `detect` | HSV thresholding + morphological cleanup | Core detection cost |
| `refine` | Median blur, open/close, dilate/erode, contour filtering | Mask refinement cost |
| `temporal` | EMA smoothing + persistence counters | Temporal stability cost |
| `render` | Alpha compositing with background | Rendering cost |

---

## 3. Per-Stage Breakdown

The benchmark measures each stage independently by inserting timing calls
around each operation:

```python
t0 = time.perf_counter()
# ... stage operation ...
t1 = time.perf_counter()
stage_times["stage_name"].append((t1 - t0) * 1000.0)
```

### Stage Contribution

The report includes a percentage breakdown showing how much each stage
contributes to the total frame latency:

```
Stage Contribution (% of total mean latency)
----------------------------------------------------------------
preprocess          2.1%  #
detect             45.3%  ####################################
refine             28.7%  ##########################
temporal            5.2%  ####
render             18.7%  ################
```

This helps identify bottlenecks. For example, if `detect` dominates, optimizing
the HSV thresholding or reducing morphological iterations would have the most
impact.

### Why Synthetic Frames?

The benchmark uses synthetic frames (a moving blue rectangle on a static
background) rather than webcam input because:

1. **Reproducibility:** Same frames every run, no camera variability
2. **No hardware dependency:** Works without a webcam or GPU
3. **Isolation:** Measures algorithm cost without I/O overhead

---

## 4. Running Benchmarks

### Default Benchmark

```bash
python benchmarks/benchmark_pipeline.py
```

### Custom Parameters

```bash
# 500 frames at 1280x720
python benchmarks/benchmark_pipeline.py --frames 500 --width 1280 --height 720

# Quick test with warmup
python benchmarks/benchmark_pipeline.py --frames 50 --warmup 5
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--frames` | 150 | Number of benchmark frames |
| `--width` | 640 | Frame width in pixels |
| `--height` | 480 | Frame height in pixels |
| `--warmup` | 10 | Warmup frames (discarded from measurements) |

### Output

The benchmark prints a formatted report to stdout:

```
========================================================================
  BLUE INVISIBILITY CLOAK — PIPELINE BENCHMARK REPORT
========================================================================

  Environment
  --------------------------------------------------------------------
  Platform        : Windows-10.0.22631-SP0
  Processor       : Intel64 Family 6 Model 142 Stepping 12 GenuineIntel
  Python          : 3.11.5
  OpenCV          : 4.8.1
  NumPy           : 1.25.2
  Resolution      : 640x480

  Configuration
  --------------------------------------------------------------------
  Frames measured : 150
  Detection mode  : HSV only

  Per-Stage Timing Breakdown (ms)
  --------------------------------------------------------------------
  Stage               Mean     Median        P95
  --------------------------------------------------------------------
  preprocess          X.XXms     X.XXms     X.XXms
  detect              X.XXms     X.XXms     X.XXms
  refine              X.XXms     X.XXms     X.XXms
  temporal            X.XXms     X.XXms     X.XXms
  render              X.XXms     X.XXms     X.XXms
  --------------------------------------------------------------------

  Total Frame Latency (ms)
  --------------------------------------------------------------------
  Mean            : X.XXms
  Median          : X.XXms
  P95             : X.XXms

  Frames Per Second
  --------------------------------------------------------------------
  Mean            : XX.X
  Median          : XX.X
  P95             : XX.X

  Stage Contribution (% of total mean latency)
  --------------------------------------------------------------------
  preprocess       X.X%  #
  detect          XX.X%  ################
  refine          XX.X%  ###############
  temporal         X.X%  ##
  render          XX.X%  ##########
========================================================================
```

---

## 5. Current Results

### Status: PENDING

No benchmark results have been recorded yet. To populate this section:

1. Run the benchmark on your target hardware:
   ```bash
   python benchmarks/benchmark_pipeline.py --frames 200 > benchmarks/results_<machine>.md
   ```

2. Record the machine specifications (CPU, RAM, Python version, OpenCV version)

3. Note the resolution and frame count used

4. Paste the results into this section

### Expected Factors

Performance will vary significantly based on:

- **CPU:** Single-threaded performance is the primary bottleneck
- **Resolution:** Doubling resolution roughly quadruples per-stage costs
- **Morphological kernel size:** Larger kernels = more expensive operations
- **Temporal smoothing:** Minimal overhead (EMA is fast)

### How to Interpret

- **Median FPS** is the most meaningful metric for real-time applications
- **P95 latency** shows worst-case frame time (important for smooth display)
- If P95 is significantly higher than median, the pipeline has inconsistent
  frame times (investigate `detect` or `refine` stages)
- Target: 30 FPS (33ms per frame) for smooth real-time display at 30 FPS webcam
