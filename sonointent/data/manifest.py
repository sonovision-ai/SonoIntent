"""Read/validate the portable public SonoIntent image index (public_id-keyed)."""
import csv, os

# Real schema of the released image_index.csv (11 fields). The 9-field version
# predates the asset-access fields; both are documented in DATASET_CARD.md.
INDEX_FIELDS = ["public_id", "dataset", "split", "ood_type",
                "asset_image", "asset_mask", "image_sha256", "mask_sha256",
                "license_tier", "asset_availability", "asset_available_now"]

def load_index(metadata_dir):
    path = os.path.join(metadata_dir, "image_index.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty index: {path}")
    if set(rows[0]) != set(INDEX_FIELDS):
        raise ValueError(
            f"unexpected index schema in {path}: got {sorted(rows[0])}, "
            f"expected {sorted(INDEX_FIELDS)}")
    return rows

def by_public_id(rows):
    return {r["public_id"]: r for r in rows}
