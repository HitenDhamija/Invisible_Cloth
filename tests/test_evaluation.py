"""
Tests for the detection quality evaluation framework.

Covers: perfect match, no overlap, partial overlap, empty masks,
full coverage, no false positives, and edge cases.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "benchmarks")
)
from evaluate_mask import MaskEvaluator, compare_masks, print_report


@pytest.fixture
def ev():
    return MaskEvaluator()


# ── 1. Perfect match ────────────────────────────────────────────────────────


class TestPerfectMatch:
    def test_iou(self, ev):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        assert ev.iou(mask, mask.copy()) == pytest.approx(1.0)

    def test_dice(self, ev):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        assert ev.dice(mask, mask.copy()) == pytest.approx(1.0)

    def test_precision(self, ev):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        assert ev.precision(mask, mask.copy()) == pytest.approx(1.0)

    def test_recall(self, ev):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        assert ev.recall(mask, mask.copy()) == pytest.approx(1.0)

    def test_all_metrics(self, ev):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        m = ev.evaluate(mask, mask.copy())
        assert m == pytest.approx({"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0})


# ── 2. No overlap ───────────────────────────────────────────────────────────


class TestNoOverlap:
    def test_iou(self, ev):
        a = np.zeros((10, 10), dtype=bool)
        a[0:3, 0:3] = True
        b = np.zeros((10, 10), dtype=bool)
        b[7:10, 7:10] = True
        assert ev.iou(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_dice(self, ev):
        a = np.zeros((10, 10), dtype=bool)
        a[0:3, 0:3] = True
        b = np.zeros((10, 10), dtype=bool)
        b[7:10, 7:10] = True
        assert ev.dice(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_precision(self, ev):
        a = np.zeros((10, 10), dtype=bool)
        a[0:3, 0:3] = True
        b = np.zeros((10, 10), dtype=bool)
        b[7:10, 7:10] = True
        assert ev.precision(a, b) == pytest.approx(0.0)

    def test_recall(self, ev):
        a = np.zeros((10, 10), dtype=bool)
        a[0:3, 0:3] = True
        b = np.zeros((10, 10), dtype=bool)
        b[7:10, 7:10] = True
        assert ev.recall(a, b) == pytest.approx(0.0)


# ── 3. Partial overlap ──────────────────────────────────────────────────────


class TestPartialOverlap:
    def test_values(self, ev):
        a = np.zeros((10, 10), dtype=bool)
        a[0:5, 0:5] = True  # 25 pixels
        b = np.zeros((10, 10), dtype=bool)
        b[3:8, 3:8] = True  # 25 pixels
        # intersection = 4 pixels (rows 3-4, cols 3-4)
        # union = 25 + 25 - 4 = 46
        assert ev.iou(a, b) == pytest.approx(4.0 / 46.0, abs=1e-5)
        assert ev.dice(a, b) == pytest.approx(8.0 / 50.0, abs=1e-5)
        assert ev.precision(a, b) == pytest.approx(4.0 / 25.0, abs=1e-5)
        assert ev.recall(a, b) == pytest.approx(4.0 / 25.0, abs=1e-5)

    def test_asymmetric(self, ev):
        a = np.zeros((10, 10), dtype=bool)
        a[0:5, 0:5] = True  # 25 pixels
        b = np.zeros((10, 10), dtype=bool)
        b[0:3, 0:3] = True  # 9 pixels, fully inside a
        # intersection = 9, union = 25
        assert ev.iou(a, b) == pytest.approx(9.0 / 25.0, abs=1e-5)
        assert ev.precision(a, b) == pytest.approx(9.0 / 25.0, abs=1e-5)
        assert ev.recall(a, b) == pytest.approx(1.0)


# ── 4. Empty predicted mask ─────────────────────────────────────────────────


class TestEmptyPredicted:
    def test_metrics(self, ev):
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        gt[0:3, 0:3] = True
        assert ev.iou(pred, gt) == pytest.approx(0.0, abs=1e-5)
        assert ev.dice(pred, gt) == pytest.approx(0.0, abs=1e-5)
        assert ev.precision(pred, gt) == pytest.approx(0.0)
        assert ev.recall(pred, gt) == pytest.approx(0.0)


# ── 5. Empty ground truth mask ──────────────────────────────────────────────


class TestEmptyGroundTruth:
    def test_metrics(self, ev):
        pred = np.zeros((10, 10), dtype=bool)
        pred[0:3, 0:3] = True
        gt = np.zeros((10, 10), dtype=bool)
        assert ev.iou(pred, gt) == pytest.approx(0.0, abs=1e-5)
        assert ev.dice(pred, gt) == pytest.approx(0.0, abs=1e-5)
        assert ev.precision(pred, gt) == pytest.approx(0.0)
        assert ev.recall(pred, gt) == pytest.approx(0.0)


# ── 6. Both empty masks ─────────────────────────────────────────────────────


class TestBothEmpty:
    def test_metrics(self, ev):
        mask = np.zeros((10, 10), dtype=bool)
        m = ev.evaluate(mask, mask)
        assert m["iou"] == pytest.approx(0.0, abs=1e-5)
        assert m["dice"] == pytest.approx(0.0, abs=1e-5)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)


# ── 7. Full coverage (recall = 1.0) ────────────────────────────────────────


class TestFullCoverage:
    def test_recall_one(self, ev):
        pred = np.zeros((10, 10), dtype=bool)
        pred[0:8, 0:8] = True  # large predicted region
        gt = np.zeros((10, 10), dtype=bool)
        gt[0:3, 0:3] = True  # gt fully inside pred
        assert ev.recall(pred, gt) == pytest.approx(1.0)
        assert ev.precision(pred, gt) == pytest.approx(9.0 / 64.0, abs=1e-5)

    def test_recall_one_reversed(self, ev):
        pred = np.zeros((10, 10), dtype=bool)
        pred[0:3, 0:3] = True
        gt = np.zeros((10, 10), dtype=bool)
        gt[0:8, 0:8] = True
        assert ev.recall(pred, gt) == pytest.approx(9.0 / 64.0, abs=1e-5)
        assert ev.precision(pred, gt) == pytest.approx(1.0)


# ── 8. No false positives (precision = 1.0) ────────────────────────────────


class TestNoFalsePositives:
    def test_precision_one(self, ev):
        pred = np.zeros((10, 10), dtype=bool)
        pred[0:3, 0:3] = True  # prediction is subset of gt
        gt = np.zeros((10, 10), dtype=bool)
        gt[0:8, 0:8] = True
        assert ev.precision(pred, gt) == pytest.approx(1.0)
        assert ev.recall(pred, gt) == pytest.approx(9.0 / 64.0, abs=1e-5)

    def test_precision_one_reversed(self, ev):
        pred = np.zeros((10, 10), dtype=bool)
        pred[0:8, 0:8] = True
        gt = np.zeros((10, 10), dtype=bool)
        gt[0:3, 0:3] = True  # gt is subset of pred
        assert ev.precision(pred, gt) == pytest.approx(9.0 / 64.0, abs=1e-5)
        assert ev.recall(pred, gt) == pytest.approx(1.0)


# ── compare_masks convenience function ──────────────────────────────────────


class TestCompareMasks:
    def test_returns_all_keys(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        m = compare_masks(a, b)
        assert set(m.keys()) == {"iou", "dice", "precision", "recall"}


# ── print_report (smoke test) ───────────────────────────────────────────────


class TestPrintReport:
    def test_runs(self, capsys):
        print_report({"iou": 0.75, "dice": 0.85, "precision": 0.9, "recall": 0.8})
        captured = capsys.readouterr()
        assert "IoU" in captured.out
        assert "Dice" in captured.out
