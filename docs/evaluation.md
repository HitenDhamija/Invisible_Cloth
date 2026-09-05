# Evaluation Framework

Documentation for the binary mask evaluation metrics and how to use the
evaluation framework.

---

## Table of Contents

1. [Metrics Overview](#1-metrics-overview)
2. [IoU (Intersection over Union)](#2-iou-intersection-over-union)
3. [Dice Coefficient](#3-dice-coefficient)
4. [Precision](#4-precision)
5. [Recall](#5-recall)
6. [Using the Evaluation Framework](#6-using-the-evaluation-framework)
7. [Current Status](#7-current-status)

---

## 1. Metrics Overview

Four standard computer vision metrics are used to evaluate binary mask quality:

| Metric | Range | Best | Penalizes |
|--------|-------|------|-----------|
| IoU | [0.0, 1.0] | 1.0 | Both false positives and false negatives |
| Dice | [0.0, 1.0] | 1.0 | False negatives less than IoU |
| Precision | [0.0, 1.0] | 1.0 | False positives only |
| Recall | [0.0, 1.0] | 1.0 | False negatives only |

All metrics are computed over the entire mask (all pixels), not per-object.

---

## 2. IoU (Intersection over Union)

Also known as the **Jaccard Index**.

### Formula

$$
\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}
$$

Where:
- $P$ = predicted mask (set of foreground pixels)
- $G$ = ground-truth mask
- TP = true positives (correctly detected pixels)
- FP = false positives (incorrectly detected pixels)
- FN = false negatives (missed pixels)

### Range and Interpretation

| IoU Score | Interpretation |
|-----------|----------------|
| 1.0 | Perfect overlap |
| > 0.7 | Excellent |
| > 0.5 | Good |
| 0.3 - 0.5 | Marginal |
| < 0.3 | Poor |

### Implementation

```python
intersection = np.logical_and(pred, gt).sum()
union = np.logical_or(pred, gt).sum()
iou = (intersection + smooth) / (union + smooth)
```

A small smoothing constant ($\epsilon = 10^{-6}$) is added to avoid division by
zero when both masks are empty.

---

## 3. Dice Coefficient

Also known as the **Sørensen-Dice Coefficient** or **F1 Score for masks**.

### Formula

$$
\text{Dice} = \frac{2|P \cap G|}{|P| + |G|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}
$$

### Relationship to IoU

$$
\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}
$$

Dice is always $\geq$ IoU for the same prediction. It is more forgiving of small
false negatives because it weights the denominator by the sum of both masks'
sizes rather than their union.

### Range and Interpretation

| Dice Score | Interpretation |
|------------|----------------|
| 1.0 | Perfect overlap |
| > 0.8 | Excellent |
| > 0.6 | Good |
| 0.4 - 0.6 | Marginal |
| < 0.4 | Poor |

### Implementation

```python
intersection = np.logical_and(pred, gt).sum()
dice = (2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
```

---

## 4. Precision

### Formula

$$
\text{Precision} = \frac{|P \cap G|}{|P|} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

### Meaning

Precision answers: **"Of all pixels I marked as cloak, how many actually are?"**

- High precision = few false positives (blue objects not on a person are not
  incorrectly detected as cloak)
- Low precision = many false positives (the detector is too liberal)

### Range and Interpretation

| Precision | Interpretation |
|-----------|----------------|
| 1.0 | Every detected pixel is correct |
| > 0.8 | Good -- few false detections |
| < 0.5 | Many false positives |

### Implementation

```python
if not pred.any():
    return 0.0
intersection = np.logical_and(pred, gt).sum()
precision = intersection / pred.sum()
```

---

## 5. Recall

### Formula

$$
\text{Recall} = \frac{|P \cap G|}{|G|} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

### Meaning

Recall answers: **"Of all the actual cloak pixels, how many did I detect?"**

- High recall = few false negatives (the entire cloth is detected)
- Low recall = many false negatives (parts of the cloth are missed, causing
  visible blue patches)

### Range and Interpretation

| Recall | Interpretation |
|--------|----------------|
| 1.0 | All cloak pixels detected |
| > 0.8 | Good -- full coverage |
| < 0.5 | Significant parts of the cloth are missed |

### Implementation

```python
if not gt.any():
    return 0.0
intersection = np.logical_and(pred, gt).sum()
recall = intersection / gt.sum()
```

---

## 6. Using the Evaluation Framework

### As a Module

```python
import numpy as np
from evaluate_mask import MaskEvaluator, compare_masks, print_report

# Create evaluator
evaluator = MaskEvaluator()

# Evaluate a single pair
metrics = evaluator.evaluate(predicted_mask, ground_truth_mask)
print(metrics)  # {'iou': 0.75, 'dice': 0.86, 'precision': 0.9, 'recall': 0.83}

# Or use the convenience function
metrics = compare_masks(predicted_mask, ground_truth_mask)
print_report(metrics)
```

### As a Script

```bash
python benchmarks/evaluate_mask.py
```

This runs a built-in demo with four test cases:
1. Perfect match (all metrics = 1.0)
2. No overlap (all metrics = 0.0)
3. Partial overlap
4. Synthetic circle data with tunable overlap

### Generating Test Data

```python
from evaluate_mask import generate_synthetic_data

# Generate a 100x100 mask pair with 60% overlap
pred, gt = generate_synthetic_data(shape=(100, 100), overlap_ratio=0.6)

evaluator = MaskEvaluator()
metrics = evaluator.evaluate(pred, gt)
print_report(metrics)
```

### Output Format

```
+------------------------------------------------+
| Metric         Score  Interpretation            |
+------------------------------------------------+
| IoU            0.6000  >0.5 good, >0.7 excellent |
| Dice           0.7500  Similar to IoU, less penalty on FN |
| Precision      1.0000  High = few false positives |
| Recall         0.6000  High = few false negatives |
+------------------------------------------------+
```

---

## 7. Current Status

### No Labeled Evaluation Dataset Exists

**Important:** This project does not currently have a labeled evaluation
dataset with ground-truth masks. The evaluation framework is implemented and
functional, but no real accuracy numbers can be reported without:

1. A curated dataset of video frames with corresponding ground-truth binary
   masks marking the blue cloth region
2. Annotations across diverse lighting conditions, cloth sizes, and
   backgrounds

### What Exists

- The `MaskEvaluator` class with IoU, Dice, Precision, and Recall
- Synthetic test data generation for framework validation
- Per-frame statistics (`DetectionStats`) measuring detected pixel count and
  ratio

### What Is Needed for Rigorous Evaluation

To produce meaningful accuracy metrics, the following would be required:

1. **Dataset collection:** Record video sequences with blue cloth under various
   conditions (lighting, camera angles, cloth sizes)
2. **Annotation:** Manually create ground-truth binary masks for sampled frames
3. **Evaluation:** Run the pipeline on annotated frames and compute metrics
4. **Reporting:** Report IoU/Dice/Precision/Recall with confidence intervals

Until such a dataset is created, any accuracy claims would be unfounded. The
framework is ready -- the data is what's missing.
