"""Schema-compatibility tests for sonointent.data.manifest (real 11-field schema)."""
import csv
import os
import pytest

from sonointent.data.manifest import INDEX_FIELDS, load_index, by_public_id

def _write_fixture(tmp_path, rows):
    p = tmp_path / "image_index.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader(); w.writerows(rows)
    return tmp_path

def _row(pid, dataset="HC18", split="test"):
    return {k: "" for k in INDEX_FIELDS} | {
        "public_id": pid, "dataset": dataset, "split": split,
        "ood_type": "ID",
        "asset_image": f"assets/hc18/img/{pid}.png",
        "asset_mask": f"assets/hc18/mask/{pid}.png",
        "image_sha256": "a" * 64, "mask_sha256": "b" * 64,
        "license_tier": "candidate",
        "asset_availability": "pending_cleared", "asset_available_now": "no"}

def test_accepts_real_11_field_schema(tmp_path):
    d = _write_fixture(tmp_path, [_row("si_0000000000000001"), _row("si_0000000000000002")])
    rows = load_index(d)
    assert len(rows) == 2
    assert by_public_id(rows)["si_0000000000000001"]["asset_available_now"] == "no"

def test_rejects_legacy_9_field_schema(tmp_path):
    p = tmp_path / "image_index.csv"
    legacy = ["public_id", "dataset", "split", "ood_type", "asset_image",
              "asset_mask", "image_sha256", "mask_sha256", "license_tier"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=legacy); w.writeheader(); w.writerow({k: "" for k in legacy})
    with pytest.raises(ValueError, match="unexpected index schema"):
        load_index(tmp_path)

def test_rejects_empty_index(tmp_path):
    open(tmp_path / "image_index.csv", "w").close()
    with pytest.raises(ValueError, match="empty index"):
        load_index(tmp_path)

def test_real_package_if_present():
    repo = os.environ.get("SONOINTENT_ANNOTATIONS")
    if not repo or not os.path.exists(os.path.join(repo, "metadata", "image_index.csv")):
        pytest.skip("SONOINTENT_ANNOTATIONS not set to a release package")
    rows = load_index(os.path.join(repo, "metadata"))
    assert len(rows) == 131584
    assert len(by_public_id(rows)) == 131584
