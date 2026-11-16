"""Tests for the aggregation evaluator (official giou/ciou definitions and
package-mode integrity accounting)."""
import json
import numpy as np
import os
import pytest
from PIL import Image

from sonointent.eval import evaluate

FIELDS = ["public_id", "dataset", "split", "ood_type", "problem", "thinking",
          "asset_image", "asset_mask", "image_sha256", "mask_sha256",
          "license_tier", "asset_availability", "asset_available_now"]

def _write_mask(path, arr):
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)

def _run_pairs(tmp_path, pairs):
    p = tmp_path / "pairs.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in pairs: f.write(json.dumps(r) + "\n")
    out = tmp_path / "metrics.json"
    return evaluate.main(["--pairs", str(p), "--out", str(out)]), out

def _make_package(tmp_path, rows):
    ann = tmp_path / "annotations"; ann.mkdir()
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    for sp in ("train", "val", "test", "target_absent"):
        fn = "target_absent.jsonl" if sp == "target_absent" else f"{sp}.jsonl"
        with open(ann / fn, "w", encoding="utf-8") as f:
            for r in by_split.get(sp, []):
                f.write(json.dumps(r) + "\n")
    return tmp_path

def _row(pid, split="test", ood="ID"):
    return {k: "" for k in FIELDS} | {
        "public_id": pid, "dataset": "HC18", "split": split, "ood_type": ood,
        "problem": "q", "thinking": "t", "asset_image": f"assets/x/img/{pid}.png",
        "asset_mask": f"assets/x/mask/{pid}.png", "image_sha256": "a" * 64,
        "mask_sha256": "b" * 64, "license_tier": "candidate",
        "asset_availability": "pending_cleared", "asset_available_now": "no"}

def test_perfect_pairs(tmp_path):
    pred = tmp_path / "p.png"; gt = tmp_path / "g.png"
    a = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    _write_mask(pred, a); _write_mask(gt, a)
    rc, out = _run_pairs(tmp_path, [{"public_id": "si_1", "split": "test",
                                     "ood_type": "ID",
                                     "pred_mask": str(pred), "gt_mask": str(gt)}])
    assert rc == 0
    m = json.load(open(out))
    assert m["n_expected"] == 1 and m["n_done"] == 1
    assert m["overall"]["dice"] == pytest.approx(1.0)
    assert m["subsets"]["ID"]["n"] == 1

def test_empty_prediction_is_zero_not_missing(tmp_path):
    pred = tmp_path / "p.png"; gt = tmp_path / "g.png"
    _write_mask(pred, np.zeros((4, 4), dtype=np.uint8))
    _write_mask(gt, np.ones((4, 4), dtype=np.uint8))
    rc, out = _run_pairs(tmp_path, [{"public_id": "si_1", "split": "test",
                                     "ood_type": "ID",
                                     "pred_mask": str(pred), "gt_mask": str(gt)}])
    assert rc == 0
    assert json.load(open(out))["overall"]["dice"] == pytest.approx(0.0)

def test_missing_prediction_marks_incomplete(tmp_path):
    gt = tmp_path / "g.png"
    _write_mask(gt, np.ones((4, 4), dtype=np.uint8))
    rc, out = _run_pairs(tmp_path, [{"public_id": "si_1", "split": "test",
                                     "ood_type": "ID",
                                     "pred_mask": str(tmp_path / "absent.png"),
                                     "gt_mask": str(gt)}])
    assert rc == 2
    assert not out.exists()
    assert (tmp_path / "metrics.json.incomplete").exists()

def test_target_absent_bucketing(tmp_path):
    pred = tmp_path / "p.png"; gt = tmp_path / "g.png"
    _write_mask(pred, np.zeros((4, 4), dtype=np.uint8))
    _write_mask(gt, np.zeros((4, 4), dtype=np.uint8))
    rc, out = _run_pairs(tmp_path, [{"public_id": "si_1", "split": "target_absent",
                                     "ood_type": "target_absent",
                                     "pred_mask": str(pred), "gt_mask": str(gt)}])
    assert rc == 0
    m = json.load(open(out))
    assert m["subsets"]["target_absent"]["n"] == 1

def test_official_giou_ciou_aggregates(tmp_path):
    """giou = mean per-sample pixel IoU; ciou = sum(inter)/sum(union).
    Sample A: 4 px perfect (iou 1, inter 4, union 4).
    Sample B: gt 100 px, pred 40 px inside gt (iou .4, inter 40, union 100).
    giou = (1 + .4)/2 = .7; ciou = 44/104."""
    pa, ga = tmp_path / "pa.png", tmp_path / "ga.png"
    pb, gb = tmp_path / "pb.png", tmp_path / "gb.png"
    a = np.ones((2, 2), dtype=np.uint8)
    _write_mask(pa, a); _write_mask(ga, a)
    big = np.ones((10, 10), dtype=np.uint8)
    part = np.zeros((10, 10), dtype=np.uint8); part[:4, :10] = 1
    _write_mask(pb, part); _write_mask(gb, big)
    rc, out = _run_pairs(tmp_path, [
        {"public_id": "si_1", "split": "test", "ood_type": "ID",
         "pred_mask": str(pa), "gt_mask": str(ga)},
        {"public_id": "si_2", "split": "test", "ood_type": "OOD_intent",
         "pred_mask": str(pb), "gt_mask": str(gb)}])
    assert rc == 0
    m = json.load(open(out))
    assert m["overall"]["giou"] == pytest.approx(0.7)
    assert m["overall"]["ciou"] == pytest.approx(44 / 104)
    # per-subset aggregation also follows the same definitions
    assert m["subsets"]["ID"]["giou"] == pytest.approx(1.0)
    assert m["subsets"]["OOD_intent"]["ciou"] == pytest.approx(0.4)

def test_package_mode_two_samples_one_prediction_regression(tmp_path):
    """Submitting 1 prediction for a 2-record benchmark must NOT produce
    full-benchmark metrics."""
    pkg = _make_package(tmp_path, [_row("si_1"), _row("si_2")])
    gt = tmp_path / "gts"; gt.mkdir()
    _write_mask(gt / "si_1.png", np.ones((4, 4), dtype=np.uint8))
    _write_mask(gt / "si_2.png", np.ones((4, 4), dtype=np.uint8))
    pred = tmp_path / "p.png"; _write_mask(pred, np.ones((4, 4), dtype=np.uint8))
    preds = tmp_path / "preds.jsonl"
    with open(preds, "w", encoding="utf-8") as f:
        f.write(json.dumps({"public_id": "si_1", "pred_mask": str(pred)}) + "\n")
    out = tmp_path / "metrics.json"
    rc = evaluate.main(["--annotations", str(pkg), "--predictions", str(preds),
                        "--gt-root", str(gt), "--split", "test",
                        "--out", str(out)])
    assert rc == 2
    assert not out.exists()
    inc = json.load(open(tmp_path / "metrics.json.incomplete"))
    assert inc["problems"][0]["kind"] == "missing"
    assert inc["problems"][0]["public_ids"] == ["si_2"]

def test_package_mode_duplicate_and_unknown(tmp_path):
    pkg = _make_package(tmp_path, [_row("si_1"), _row("si_2")])
    gt = tmp_path / "gts"; gt.mkdir()
    _write_mask(gt / "si_1.png", np.ones((4, 4), dtype=np.uint8))
    _write_mask(gt / "si_2.png", np.ones((4, 4), dtype=np.uint8))
    pred = tmp_path / "p.png"; _write_mask(pred, np.ones((4, 4), dtype=np.uint8))
    preds = tmp_path / "preds.jsonl"
    with open(preds, "w", encoding="utf-8") as f:
        f.write(json.dumps({"public_id": "si_1", "pred_mask": str(pred)}) + "\n")
        f.write(json.dumps({"public_id": "si_1", "pred_mask": str(pred)}) + "\n")
        f.write(json.dumps({"public_id": "si_zzz", "pred_mask": str(pred)}) + "\n")
    out = tmp_path / "metrics.json"
    rc = evaluate.main(["--annotations", str(pkg), "--predictions", str(preds),
                        "--gt-root", str(gt), "--split", "test", "--out", str(out)])
    assert rc == 2
    assert not out.exists()
    inc = json.load(open(tmp_path / "metrics.json.incomplete"))
    kinds = {p["kind"] for p in inc["problems"]}
    assert kinds == {"missing", "duplicate", "unknown"}

def test_package_mode_complete(tmp_path):
    pkg = _make_package(tmp_path, [_row("si_1"), _row("si_2")])
    gt = tmp_path / "gts"; gt.mkdir()
    for pid in ("si_1", "si_2"):
        _write_mask(gt / f"{pid}.png", np.ones((4, 4), dtype=np.uint8))
    pred = tmp_path / "p.png"; _write_mask(pred, np.ones((4, 4), dtype=np.uint8))
    preds = tmp_path / "preds.jsonl"
    with open(preds, "w", encoding="utf-8") as f:
        for pid in ("si_1", "si_2"):
            f.write(json.dumps({"public_id": pid, "pred_mask": str(pred)}) + "\n")
    out = tmp_path / "metrics.json"
    rc = evaluate.main(["--annotations", str(pkg), "--predictions", str(preds),
                        "--gt-root", str(gt), "--split", "test", "--out", str(out)])
    assert rc == 0
    m = json.load(open(out))
    assert m["n_expected"] == 2 and m["n_done"] == 2
    assert m["overall"]["dice"] == pytest.approx(1.0)
