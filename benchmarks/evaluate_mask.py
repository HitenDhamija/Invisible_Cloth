"""
Detection Quality Evaluation Framework for Binary Masks.

Compares predicted binary masks against ground-truth masks using standard
computer vision metrics: IoU, Dice, Precision, and Recall.

Usage as module:
    from evaluate_mask import MaskEvaluator, compare_masks, print_report

    evaluator = MaskEvaluator()
    metrics = evaluator.evaluate(predicted_mask, ground_truth_mask)
    print_report(metrics)

Usage as script:
    python evaluate_mask.py
"""


import numpy as np


class MaskEvaluator:
    """Evaluates binary masks using IoU, Dice, Precision, and Recall."""

    def __init__(self, smooth: float = 1e-6):
        """Initialize evaluator.

        Args:
            smooth: Small constant to avoid division by zero.
        """
        self.smooth = smooth

    def iou(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """Intersection over Union (Jaccard Index).

        Formula: |predicted ∩ ground_truth| / |predicted ∪ ground_truth|

        Args:
            predicted: Binary mask (0/1 or bool).
            ground_truth: Binary mask (0/1 or bool).

        Returns:
            IoU score in [0.0, 1.0]. >0.5 is generally good, >0.7 is excellent.
        """
        pred = self._to_bool(predicted)
        gt = self._to_bool(ground_truth)

        if not pred.any() and not gt.any():
            return 0.0

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        return float((intersection + self.smooth) / (union + self.smooth))

    def dice(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """Dice Coefficient (F1 Score for masks).

        Formula: 2|predicted ∩ ground_truth| / (|predicted| + |ground_truth|)

        Args:
            predicted: Binary mask (0/1 or bool).
            ground_truth: Binary mask (0/1 or bool).

        Returns:
            Dice score in [0.0, 1.0]. Similar to IoU but penalizes false
            negatives less.
        """
        pred = self._to_bool(predicted)
        gt = self._to_bool(ground_truth)

        if not pred.any() and not gt.any():
            return 0.0

        intersection = np.logical_and(pred, gt).sum()

        return float(
            (2.0 * intersection + self.smooth) / (pred.sum() + gt.sum() + self.smooth)
        )

    def precision(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """Precision: fraction of predicted mask pixels that are correct.

        Formula: |predicted ∩ ground_truth| / |predicted|

        Args:
            predicted: Binary mask (0/1 or bool).
            ground_truth: Binary mask (0/1 or bool).

        Returns:
            Precision in [0.0, 1.0]. High precision = few false positives.
        """
        pred = self._to_bool(predicted)
        gt = self._to_bool(ground_truth)

        if not pred.any():
            return 0.0

        intersection = np.logical_and(pred, gt).sum()
        return float(intersection / pred.sum())

    def recall(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """Recall: fraction of ground truth mask pixels detected.

        Formula: |predicted ∩ ground_truth| / |ground_truth|

        Args:
            predicted: Binary mask (0/1 or bool).
            ground_truth: Binary mask (0/1 or bool).

        Returns:
            Recall in [0.0, 1.0]. High recall = few false negatives.
        """
        pred = self._to_bool(predicted)
        gt = self._to_bool(ground_truth)

        if not gt.any():
            return 0.0

        intersection = np.logical_and(pred, gt).sum()
        return float(intersection / gt.sum())

    def evaluate(
        self, predicted: np.ndarray, ground_truth: np.ndarray
    ) -> dict[str, float]:
        """Compute all metrics at once.

        Args:
            predicted: Binary mask (0/1 or bool).
            ground_truth: Binary mask (0/1 or bool).

        Returns:
            Dictionary with keys: iou, dice, precision, recall.
        """
        return {
            "iou": self.iou(predicted, ground_truth),
            "dice": self.dice(predicted, ground_truth),
            "precision": self.precision(predicted, ground_truth),
            "recall": self.recall(predicted, ground_truth),
        }

    @staticmethod
    def _to_bool(mask: np.ndarray) -> np.ndarray:
        """Convert mask to boolean array."""
        return mask.astype(bool)


def compare_masks(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> dict[str, float]:
    """Compare two binary masks and return all metrics.

    Args:
        predicted: Binary mask (0/1 or bool).
        ground_truth: Binary mask (0/1 or bool).

    Returns:
        Dictionary with keys: iou, dice, precision, recall.

    Example:
        >>> pred = np.array([[1,1,0],[1,0,0]])
        >>> gt   = np.array([[1,1,1],[1,0,0]])
        >>> metrics = compare_masks(pred, gt)
        >>> print(metrics['iou'])
        0.6
    """
    evaluator = MaskEvaluator()
    return evaluator.evaluate(predicted, ground_truth)


def print_report(metrics: dict[str, float]) -> None:
    """Print a formatted evaluation report.

    Args:
        metrics: Dictionary from compare_masks() or MaskEvaluator.evaluate().

    Example:
        >>> metrics = compare_masks(pred, gt)
        >>> print_report(metrics)
        ┌──────────────┬───────────┐
        │ Metric       │   Score   │
        ├──────────────┼───────────┤
        │ IoU          │     0.600 │
        │ Dice         │     0.750 │
        │ Precision    │     1.000 │
        │ Recall       │     0.600 │
        └──────────────┴───────────┘
    """
    labels = {
        "iou": "IoU",
        "dice": "Dice",
        "precision": "Precision",
        "recall": "Recall",
    }
    interpretations = {
        "iou": ">0.5 good, >0.7 excellent",
        "dice": "Similar to IoU, less penalty on FN",
        "precision": "High = few false positives",
        "recall": "High = few false negatives",
    }

    header = f"{'Metric':<14} {'Score':>9}  {'Interpretation'}"
    sep = "-" * len(header)
    print(f"+{sep}+")
    print(f"| {header} |")
    print(f"+{sep}+")
    for key, label in labels.items():
        score = metrics.get(key, 0.0)
        interp = interpretations[key]
        print(f"| {label:<14} {score:>9.4f}  {interp} |")
    print(f"+{sep}+")


def generate_synthetic_data(
    shape: tuple[int, int] = (100, 100),
    overlap_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic mask pair for demonstration.

    Creates a ground-truth circle and a predicted circle with tunable overlap.

    Args:
        shape: Height and width of the mask.
        overlap_ratio: Fraction of predicted circle that overlaps ground truth
            (0.0 = no overlap, 1.0 = perfect match).
    seed: Random seed for reproducibility.

    Returns:
        Tuple of (predicted_mask, ground_truth_mask).
    """
    h, w = shape
    gt = np.zeros(shape, dtype=bool)

    cy_gt, cx_gt = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius_gt = min(h, w) // 4
    gt_mask = (yy - cy_gt) ** 2 + (xx - cx_gt) ** 2 <= radius_gt ** 2
    gt[gt_mask] = True

    pred = np.zeros(shape, dtype=bool)
    max_offset = radius_gt
    offset_x = int(max_offset * (1.0 - overlap_ratio))
    cy_pred, cx_pred = cy_gt, cx_gt + offset_x
    pred_mask = (yy - cy_pred) ** 2 + (xx - cx_pred) ** 2 <= radius_gt ** 2
    pred[pred_mask] = True

    return pred, gt


if __name__ == "__main__":
    print("=== Detection Quality Evaluation Framework — Demo ===\n")

    evaluator = MaskEvaluator()

    # 1. Perfect match
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    print("1. Perfect match:")
    m = evaluator.evaluate(mask, mask.copy())
    print_report(m)
    print()

    # 2. No overlap
    a = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool)
    b[7:10, 7:10] = True
    print("2. No overlap:")
    print_report(evaluator.evaluate(a, b))
    print()

    # 3. Partial overlap
    a = np.zeros((10, 10), dtype=bool)
    a[0:5, 0:5] = True
    b = np.zeros((10, 10), dtype=bool)
    b[3:8, 3:8] = True
    print("3. Partial overlap:")
    print_report(evaluator.evaluate(a, b))
    print()

    # 4. Synthetic circle data
    pred, gt = generate_synthetic_data((100, 100), overlap_ratio=0.6)
    print("4. Synthetic circle data (60% overlap):")
    print_report(evaluator.evaluate(pred, gt))
