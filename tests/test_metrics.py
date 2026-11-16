"""Unit tests for sonointent.eval.metrics against the official definition."""
import numpy as np
import pytest

from sonointent.eval.metrics import score, dice, iou

def test_perfect_match_uint8_255():
    p = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    d, io, pr, rc, inter, union = score(p, p.copy())
    assert (d, io, pr, rc) == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert inter == pytest.approx(2.0) and union == pytest.approx(2.0)

def test_partial_overlap():
    p = np.array([1, 1, 0, 0], dtype=np.uint8)
    g = np.array([1, 0, 1, 0], dtype=np.uint8)
    d, io, pr, rc, _, _ = score(p, g)
    assert d == pytest.approx(0.5)
    assert io == pytest.approx(1 / 3)
    assert pr == pytest.approx(0.5)
    assert rc == pytest.approx(0.5)

def test_disjoint():
    p = np.array([1, 0], dtype=np.uint8)
    g = np.array([0, 1], dtype=np.uint8)
    assert score(p, g)[:4] == pytest.approx((0.0, 0.0, 0.0, 0.0))

def test_both_empty_scores_zero():
    p = np.zeros((4, 4), dtype=np.uint8)
    g = np.zeros((4, 4), dtype=np.uint8)
    assert score(p, g)[:4] == pytest.approx((0.0, 0.0, 0.0, 0.0))

def test_pred_empty_against_target_scores_zero():
    g = np.ones((4, 4), dtype=np.uint8)
    d, io, pr, rc, _, _ = score(np.zeros_like(g), g)
    assert (d, io, pr) == pytest.approx((0.0, 0.0, 0.0))
    assert rc == pytest.approx(0.0)

def test_target_empty_against_pred_scores_zero():
    p = np.ones((4, 4), dtype=np.uint8)
    d, io, pr, rc, _, _ = score(p, np.zeros_like(p))
    assert (d, io, pr, rc) == pytest.approx((0.0, 0.0, 0.0, 0.0))

def test_bool_masks():
    p = np.array([[True, False], [False, True]])
    assert dice(p, p) == pytest.approx(1.0)

def test_01_masks_thresholded_above_zero():
    p = np.array([0.9, 0.4, 0.0, 0.2], dtype=np.float32)
    g = np.array([1, 0, 0, 1], dtype=np.float32)
    # bin: p=[1,1,0,1] (sp=3), g=[1,0,0,1] (sg=2), inter=2 -> dice=0.8
    assert dice(p, g) == pytest.approx(0.8)

def test_float_prediction_against_255_gt():
    p = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    g = np.array([[0, 255], [255, 255]], dtype=np.uint8)
    d, io, pr, rc, _, _ = score(p, g)
    assert (d, io, pr, rc) == pytest.approx((1.0, 1.0, 1.0, 1.0))

def test_wrappers():
    p = np.eye(3, dtype=np.uint8)
    assert dice(p, p) == pytest.approx(1.0)
    assert iou(p, p) == pytest.approx(1.0)
