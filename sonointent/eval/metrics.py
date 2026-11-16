"""SonoIntent official per-sample segmentation metrics.

Mirrors the established scoring definition (exps/code/common/ours_test_lib.py
`score()` in the frozen workspace):

  * inputs are binarized first — bool arrays as-is; float arrays in [0,1]
    thresholded at ``> 0`` (the official post-processing rule); integer masks
    with values above 1 (e.g. 0/255 PNGs) thresholded at ``> 127``;
  * ``dice  = 2*inter / (sum(pred) + sum(gt) + 1e-7)``
  * ``iou   = inter / (union + 1e-7)``
  * ``prec  = inter / (sum(pred) + 1e-7)``
  * ``rec   = inter / (sum(gt) + 1e-7)``

Empty-handling follows the official definition with **no special casing**:
an empty prediction against a non-empty target scores 0 (a real model
failure); two empty masks also score 0 on dice/iou (inter = 0, eps-only
denominator). One-sided-empty cases yield 0 everywhere except precision/recall
which follow the same epsilon denominators (0 for the empty side).
"""
import numpy as np

_EPS = 1e-7

def _bin(a):
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return a.astype(np.uint8)
    a = a.astype(np.float32)
    if float(a.max(initial=0)) <= 1.0:
        return (a > 0).astype(np.uint8)      # probabilities / 0-1 logits
    return (a > 127).astype(np.uint8)        # 0/255 stored masks

def score(pred, gt):
    """Official per-sample tuple: (dice, iou, precision, recall, inter, union).

    pred/gt are arrays of identical shape; pred may be float (thresholded at
    ``> 0``), gt is typically a 0/255 mask."""
    p = _bin(pred).reshape(-1)
    g = _bin(gt).reshape(-1)
    sp, sg = float(p.sum()), float(g.sum())
    inter = float((p * g).sum())
    union = sp + sg - inter
    dice = 2 * inter / (sp + sg + _EPS)
    iou = inter / (union + _EPS)
    prec = inter / (sp + _EPS)
    rec = inter / (sg + _EPS)
    return dice, iou, prec, rec, inter, union

def dice(pred, gt):
    return score(pred, gt)[0]

def iou(pred, gt):
    return score(pred, gt)[1]
