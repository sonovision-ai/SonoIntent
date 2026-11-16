"""Aggregate evaluation for SonoIntent predictions.

Two input modes, both read-only over user-provided files:

  pairs mode (self-contained; runs without the dataset package):
    --pairs pairs.jsonl   lines: {"public_id", "split", "ood_type",
                                  "pred_mask", "gt_mask"}
    (split/ood_type may be omitted; then --annotations must be given)

  package mode (needs the public SonoIntent-Annotations package for the
  expected public_id set and per-record labels, plus GT masks provided by
  the user):
    --annotations <dir> --predictions preds.jsonl --gt-root <dir>
        [--split test] [--ood-type OOD_intent]
    preds.jsonl lines: {"public_id", "pred_mask"}
    GT files live at <gt-root>/<public_id>.png

Integrity (package mode): the FULL expected public_id set of the requested
split (optionally filtered by --ood-type) is the benchmark. Missing
predictions, duplicate public_ids, unknown public_ids and unreadable masks
all mark the run INCOMPLETE — no metrics are written and the exit code is 2.
A partial submission can never produce full-benchmark numbers.

Official aggregate definitions (mirrors the paper's evaluation pipeline):
    dice = mean of per-sample Dice
    giou = mean of per-sample pixel IoU
    ciou = sum(pixel intersection) / sum(pixel union)

Output metrics.json:
  {n_expected, n_done, overall: {dice, giou, ciou},
   subsets: {<subset>: {n, dice, giou, ciou}}, incomplete: [...]}

Exit codes: 0 complete, 2 incomplete.
"""
import argparse, json, os, sys

import numpy as np

from .metrics import _bin
from ..data.annotations import iter_all

SUBSETS = ["ID", "OOD_dataset", "OOD_organ", "OOD_intent", "target_absent"]

def _load_mask(path):
    """0/255 or 0/1 or bool image -> uint8 binary array."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"))
    return _bin(a)

def _agg(rows):
    """Official aggregates. rows: per-sample dicts with dice/oiu/inter/union."""
    n = len(rows)
    inter = sum(r["inter"] for r in rows)
    union = sum(r["union"] for r in rows)
    return {"dice": float(np.mean([r["dice"] for r in rows])) if n else float("nan"),
            "giou": float(np.mean([r["iou"] for r in rows])) if n else float("nan"),
            "ciou": float(inter / (union + 1e-7)) if n else float("nan")}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--gt-root", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ood-type", default=None)
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args(argv)

    records = []
    if args.pairs:
        meta = None
        if args.annotations is not None:
            meta = {x["public_id"]: x for x in iter_all(args.annotations)}
        for line in open(args.pairs, encoding="utf-8"):
            r = json.loads(line)
            if "split" not in r or "ood_type" not in r:
                if meta is None:
                    print("--pairs rows without split/ood_type require --annotations")
                    return 2
                m = meta[r["public_id"]]
                r.setdefault("split", m["split"])
                r.setdefault("ood_type", m["ood_type"])
            records.append(r)
    else:
        if not (args.annotations and args.predictions and args.gt_root):
            print("need --pairs, or (--annotations --predictions --gt-root)")
            return 2
        meta = {x["public_id"]: x for x in iter_all(args.annotations)}
        expected = {pid for pid, m in meta.items() if m["split"] == args.split}
        if args.ood_type is not None:
            expected = {pid for pid in expected if meta[pid]["ood_type"] == args.ood_type}
        if not expected:
            print(f"no expected records for split={args.split} ood_type={args.ood_type}")
            return 2
        preds, seen, dup, unknown = {}, set(), [], []
        for line in open(args.predictions, encoding="utf-8"):
            pr = json.loads(line)
            pid = pr["public_id"]
            if pid in seen: dup.append(pid)
            seen.add(pid)
            preds[pid] = pr
        unknown = sorted(set(preds) - expected)
        missing = sorted(expected - set(preds))
        problems = []
        if missing: problems.append({"kind": "missing", "public_ids": missing})
        if dup: problems.append({"kind": "duplicate", "public_ids": dup})
        if unknown: problems.append({"kind": "unknown", "public_ids": unknown})
        if problems:
            print(f"INCOMPLETE: integrity problems: "
                  f"missing={len(missing)} duplicate={len(dup)} unknown={len(unknown)}")
            with open(args.out + ".incomplete", "w", encoding="utf-8") as f:
                json.dump({"problems": problems}, f, indent=2)
            return 2
        for pid in sorted(expected):
            pr = preds[pid]
            m = meta[pid]
            records.append({"public_id": pid, "split": args.split,
                            "ood_type": m["ood_type"], "pred_mask": pr["pred_mask"],
                            "gt_mask": os.path.join(args.gt_root, pid + ".png")})

    if not records:
        print("no records"); return 2

    incomplete = []
    per_subset = {"Overall": []}
    for s in SUBSETS:
        per_subset[s] = []
    for r in records:
        try:
            p = _load_mask(r["pred_mask"]); g = _load_mask(r["gt_mask"])
        except Exception as e:
            incomplete.append({"public_id": r["public_id"], "error": str(e)})
            continue
        sp = float(p.sum()); sg = float(g.sum())
        inter = float((p & g).sum()); union = sp + sg - inter
        d = 2 * inter / (sp + sg + 1e-7)
        iou = inter / (union + 1e-7)
        subset = r["ood_type"] if r["split"] == "test" and r["ood_type"] in SUBSETS \
            else ("target_absent" if r["split"] == "target_absent" else None)
        entry = {"dice": d, "iou": iou, "inter": inter, "union": union}
        per_subset["Overall"].append(entry)
        if subset is not None:
            per_subset[subset].append(entry)

    out = {"n_expected": len(records), "n_done": len(records) - len(incomplete),
           "overall": _agg(per_subset["Overall"]),
           "subsets": {s: {"n": len(v), **_agg(v)} for s, v in per_subset.items()
                       if s != "Overall" and v},
           "incomplete": incomplete}
    if incomplete:
        print(f"INCOMPLETE: {len(incomplete)}/{len(records)} records failed "
              f"(details in {args.out}.incomplete — no metrics written)")
        with open(args.out + ".incomplete", "w", encoding="utf-8") as f:
            json.dump(incomplete, f, indent=2)
        return 2
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"complete: {len(records)} records -> {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
