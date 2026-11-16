# -*- coding: utf-8 -*-
"""TargetAcc and TA-Dice from per-row target labels.

Definitions follow the paper's evaluation pipeline:

    TargetAcc = n_correct / n_evaluable
    TA-Dice   = sum(dice * 1[correct]) / n_evaluable
    mean Dice = sum(dice) / n_total

Rows labelled `C_unverifiable` are excluded from numerator and denominator;
`A_unstated` stays in the denominator with zero contribution; `incorrect`
contributes zero.

A row is a mapping with `model`, `setting`, `label_final` and `dice`.
`setting` is one of `ID`, `Dataset-OOD`, `Objective-OOD`, `Organ-OOD`; every
row also contributes to `Overall`.
"""
import csv
from collections import defaultdict

LABELS = {"correct", "incorrect", "A_unstated", "C_unverifiable"}
GROUPS = ("Overall", "ID", "Dataset-OOD", "Objective-OOD", "Organ-OOD")


def aggregate(rows, models=None):
    """Return one metrics record per (model, group).

    `rows` is any iterable of per-sample records; `models` fixes the output
    order and defaults to the order of first appearance.
    """
    agg = defaultdict(lambda: defaultdict(float))
    order = []
    for r in rows:
        m = r["model"]
        if m not in order:
            order.append(m)
        label = r["label_final"]
        if label not in LABELS:
            raise ValueError(f"unknown label: {label!r}")
        for grp in ("Overall", r["setting"]):
            a = agg[(grp, m)]
            a["total"] += 1
            a["dice_sum"] += float(r["dice"])
            if label == "C_unverifiable":
                a["unverifiable"] += 1
                continue
            a["evaluable"] += 1
            if label == "correct":
                a["correct"] += 1
                a["ta_dice_sum"] += float(r["dice"])
            elif label == "incorrect":
                a["incorrect"] += 1
            else:
                a["unstated"] += 1

    out = []
    for m in (models or order):
        for grp in GROUPS:
            a = agg[(grp, m)]
            ev = int(a["evaluable"])
            out.append({
                "model": m, "setting": grp,
                "total": int(a["total"]), "evaluable": ev,
                "correct": int(a["correct"]), "incorrect": int(a["incorrect"]),
                "unstated": int(a["unstated"]),
                "unverifiable": int(a["unverifiable"]),
                "TargetAcc": None if ev == 0 else a["correct"] / ev,
                "TA_Dice": None if ev == 0 else a["ta_dice_sum"] / ev,
                "mean_dice": None if not a["total"] else a["dice_sum"] / a["total"],
            })
    return out


def write_tsv(results, path):
    cols = ["model", "setting", "total", "evaluable", "correct", "incorrect",
            "unstated", "unverifiable", "TargetAcc", "TA_Dice", "mean_dice"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in results:
            w.writerow([
                r["model"], r["setting"], r["total"], r["evaluable"],
                r["correct"], r["incorrect"], r["unstated"], r["unverifiable"],
                "" if r["TargetAcc"] is None else f"{r['TargetAcc']:.12f}",
                "" if r["TA_Dice"] is None else f"{r['TA_Dice']:.12f}",
                "" if r["mean_dice"] is None else f"{r['mean_dice']:.12f}",
            ])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True,
                    help="TSV with model, setting, label_final, dice")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.rows, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    write_tsv(aggregate(rows), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
