# -*- coding: utf-8 -*-
"""Official SonoIntent split: held-out OOD sources plus an ID 90/5/5 split.

OOD test sets are whole datasets. The remaining (ID) datasets are split
90/5/5 by the highest available group level -- patient, video, sweep, case or
volume -- so that no subject crosses train, val and test. Seed 42.

The input is the rendered V1 record file; each row must carry `dataset`,
`sample_id`, `rendered_image` and `rendered_mask`.
"""
import argparse
import collections
import json
import os
import random
import re

OOD_DS = {"BUS-UC": "OOD_dataset", "DDTI": "OOD_dataset",
          "MUP + MicroSegNet": "OOD_dataset", "MMOTU-3D": "OOD_dataset",
          "Fast-U-Net": "OOD_dataset"}
OOD_INTENT = {"TG3K": "OOD_intent", "Segthy": "OOD_intent", "CUBS": "OOD_intent"}
OOD_ORGAN = {"STMUS NDA": "OOD_organ"}


def group_of(r):
    """Highest group identifier available for a record."""
    ds, sid = r["dataset"], r["sample_id"]
    if ds == "CAMUS":
        parts = sid.split("__")[0].split("_")
        return "patient:" + parts[1]
    if ds == "EchoNet-Dynamic":
        parts = sid.split("_")
        return "vid:" + "_".join(parts[1:-1])
    if ds == "CardiacUDC":
        base = sid.split("__")[0]
        m = re.search(r"_(f\d+)$", base)
        return "vid:" + base[:m.start()].split("_", 1)[1]
    if ds == "ACOUSLIC-AI":
        return "sweep:" + sid.split("_")[1]
    if ds == "FASS":
        return "patient:" + sid.split("__")[0].split("_")[1].split("_")[0]
    if ds == "FH-PS-AOP (PSFHS)":
        return "case:" + sid.split("__")[0].split("_", 1)[1]
    if ds == "regPro (μ-RegPro)":
        parts = sid.split("_")
        return "vol:" + "_".join(parts[1:-1])
    if ds == "BrEaST":
        return "case:" + sid.split("case")[1]
    return "img:" + sid


def assign(rows, seed=42):
    """Return (train, val, test) record lists."""
    train, val, test = [], [], []
    for r in rows:
        ds = r["dataset"]
        if ds in OOD_DS:
            test.append(dict(r, ood_type=OOD_DS[ds]))
        elif ds in OOD_INTENT:
            test.append(dict(r, ood_type=OOD_INTENT[ds]))
        elif ds in OOD_ORGAN:
            test.append(dict(r, ood_type=OOD_ORGAN[ds]))

    id_pool = [r for r in rows if r["dataset"] not in OOD_DS
               and r["dataset"] not in OOD_INTENT
               and r["dataset"] not in OOD_ORGAN]
    by_ds = collections.defaultdict(list)
    for r in id_pool:
        by_ds[r["dataset"]].append(r)

    rng = random.Random(seed)
    for _, pool in sorted(by_ds.items()):
        groups = collections.defaultdict(list)
        for r in pool:
            groups[group_of(r)].append(r)
        gkeys = list(groups)
        rng.shuffle(gkeys)
        total = len(pool)
        t_n = v_n = 0
        t_budget = int(round(total * 0.90))
        v_budget = int(round(total * 0.05))
        for g in gkeys:
            n = len(groups[g])
            if t_n < t_budget:
                train.extend(groups[g])
                t_n += n
            elif v_n < v_budget:
                val.extend(groups[g])
                v_n += n
            else:
                test.extend(dict(r, ood_type="ID") for r in groups[g])
    return train, val, test


def _dump(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_split(rows, out_dir, seed=42, check_files=True):
    """Write train/val/test jsonl and return composition counts."""
    train, val, test = assign(rows, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    _dump(os.path.join(out_dir, "train.jsonl"), train)
    _dump(os.path.join(out_dir, "val.jsonl"), val)
    _dump(os.path.join(out_dir, "test.jsonl"), test)

    counts = collections.Counter(r["ood_type"] for r in test)
    counts["ID"] = sum(1 for r in test if r["ood_type"] == "ID")
    n_ood = sum(1 for r in train + val
                if r["dataset"] in set(OOD_DS) | set(OOD_INTENT) | set(OOD_ORGAN))
    sids = [set(x["sample_id"] for x in part) for part in (train, val, test)]
    disjoint = not (sids[0] & sids[1] or sids[0] & sids[2] or sids[1] & sids[2])
    missing = 0
    if check_files:
        root = os.path.dirname(os.path.abspath(out_dir))
        for part in (train, val, test):
            for r in part:
                for k in ("rendered_image", "rendered_mask"):
                    if not os.path.exists(os.path.join(root, r[k])):
                        missing += 1
    return {
        "train": len(train), "val": len(val), "test": len(test),
        "ID": counts["ID"], "OOD_dataset": counts["OOD_dataset"],
        "OOD_intent": counts["OOD_intent"], "OOD_organ": counts["OOD_organ"],
        "disjoint": disjoint, "ood_in_train_or_val": n_ood,
        "missing_files": missing,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="rendered V1 records (train_full_v1_rendered.jsonl)")
    ap.add_argument("--out-dir", required=True,
                    help="output directory for train/val/test.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    with open(a.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    stats = write_split(rows, a.out_dir, seed=a.seed)
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0 if stats["disjoint"] and stats["ood_in_train_or_val"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
