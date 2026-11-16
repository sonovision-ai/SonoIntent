"""Tests for the public annotation-package loader (text-only)."""
import json
import os
import pytest

from sonointent.data.annotations import load_split, iter_all, SPLITS

FIELDS = ["public_id", "dataset", "split", "ood_type", "problem", "thinking",
          "asset_image", "asset_mask", "image_sha256", "mask_sha256",
          "license_tier", "asset_availability", "asset_available_now"]

def _make_package(tmp_path, counts=None):
    counts = counts or {"train": 2, "val": 1, "test": 1, "target_absent": 1}
    ann = tmp_path / "annotations"; ann.mkdir()
    n = 0
    for sp in SPLITS:
        rows = []
        for i in range(counts[sp]):
            n += 1
            rows.append({k: "" for k in FIELDS} | {
                "public_id": f"si_{n:016x}", "dataset": "HC18", "split": sp,
                "ood_type": "ID" if sp == "test" else
                            ("target_absent" if sp == "target_absent" else ""),
                "problem": "q", "thinking": "t" if sp != "target_absent" else "",
                "asset_image": f"assets/hc18/img/si_{n:016x}.png",
                "asset_mask": f"assets/hc18/mask/si_{n:016x}.png",
                "image_sha256": "a" * 64, "mask_sha256": "b" * 64,
                "license_tier": "candidate",
                "asset_availability": "pending_cleared", "asset_available_now": "no"})
        fn = "target_absent.jsonl" if sp == "target_absent" else f"{sp}.jsonl"
        with open(ann / fn, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    return tmp_path

def test_load_split_counts(tmp_path):
    pkg = _make_package(tmp_path)
    assert len(load_split("train", pkg)) == 2
    assert len(load_split("target_absent", pkg)) == 1

def test_iter_all_total(tmp_path):
    pkg = _make_package(tmp_path)
    assert sum(1 for _ in iter_all(pkg)) == 5

def test_split_field_forced(tmp_path):
    pkg = _make_package(tmp_path)
    rows = load_split("val", pkg)
    assert all(r["split"] == "val" for r in rows)

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_all(tmp_path))
