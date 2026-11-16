"""Load the public SonoIntent-Annotations package (text-only, no pixels).

The package is a directory with:

    annotations/{train,val,test,target_absent}.jsonl
    metadata/image_index.csv

Each JSONL line carries the public fields (public_id, dataset, split,
ood_type, problem, thinking, asset_image, asset_mask, image_sha256,
mask_sha256, license_tier, asset_availability, asset_available_now).
See the dataset card shipped with the package for field semantics.

Example:
    from sonointent.data.annotations import load_split, iter_all
    test = load_split("test", "SonoIntent-Annotations")   # list of dicts
    n = sum(1 for _ in iter_all("SonoIntent-Annotations"))  # 131,584
"""
import json, os

SPLITS = ("train", "val", "test", "target_absent")

def _path(annotations_dir, split):
    fn = "target_absent.jsonl" if split == "target_absent" else f"{split}.jsonl"
    return os.path.join(annotations_dir, "annotations", fn)

def iter_split(split, annotations_dir):
    p = _path(annotations_dir, split)
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing annotation file: {p}")
    with open(p, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["split"] = split
            yield row

def load_split(split, annotations_dir):
    return list(iter_split(split, annotations_dir))

def iter_all(annotations_dir):
    for sp in SPLITS:
        yield from iter_split(sp, annotations_dir)
