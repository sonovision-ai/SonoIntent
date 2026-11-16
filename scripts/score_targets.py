#!/usr/bin/env python3
"""Compute TargetAcc and TA-Dice from a SonoIntent test split and predictions.

Inputs
  --split        test.jsonl (or another split of the same schema)
  --predictions  predictions.jsonl produced by the evaluation harness

Each split row supplies the canonical target via its `rendered_mask` filename;
each prediction supplies `generated_text` and `dice`. The target named in the
generated reasoning is resolved with the deterministic alias table in
`sonointent.eval.target_accuracy`. Rows with no resolvable target are counted
as `A_unstated`; rows without generated text are `C_unverifiable` and are
excluded from the metric.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonointent.eval.target_accuracy import ZH_SLUG, label_for_text, target_of
from sonointent.eval.target_metrics import aggregate, write_tsv

SETTING = {"ID": "ID", "OOD_dataset": "Dataset-OOD",
           "OOD_intent": "Objective-OOD", "OOD_organ": "Organ-OOD"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--model", default="model")
    ap.add_argument("--out", required=True, help="output TSV")
    a = ap.parse_args(argv)

    split = [json.loads(l) for l in open(a.split, encoding="utf-8") if l.strip()]
    preds = {}
    with open(a.predictions, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[r["sample_id"]] = r
    missing = [r["sample_id"] for r in split if r["sample_id"] not in preds]
    if missing:
        print(f"{len(missing)} split rows have no prediction; first: {missing[:3]}",
              file=sys.stderr)
        return 2

    rows = []
    for s in split:
        p = preds[s["sample_id"]]
        canon = target_of(s)
        label = label_for_text(p.get("generated_text"),
                               ZH_SLUG.get(canon, canon))
        rows.append({
            "model": a.model,
            "setting": SETTING[s["ood_type"]],
            "label_final": label,
            "dice": float(p.get("dice") or 0.0),
        })

    results = aggregate(rows, models=[a.model])
    write_tsv(results, a.out)
    for r in results:
        if r["setting"] in ("Overall", "Objective-OOD") and r["evaluable"]:
            print(f"{r['setting']}: n={r['total']} TargetAcc={r['TargetAcc']:.4f} "
                  f"TA-Dice={r['TA_Dice']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
