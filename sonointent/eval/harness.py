# -*- coding: utf-8 -*-
"""Model-agnostic evaluation harness for the SonoIntent test split.

Every model family drives this module so that all runs produce the same
`metrics.json` / `paper_results.md` tables and the same failure accounting.

Protocol: end-to-end rendered image and question only, greedy decoding,
`max_new_tokens=384`, post-processing threshold `pred > 0`, Dice / gIoU /
cIoU over ID, Dataset-OOD, Objective-OOD and Anatomy-OOD.

A sample that raises is not a dice-0 sample. It is retried once; if it still
fails, the run is marked INCOMPLETE, no metrics are written, and the process
exits 2. A missing result never silently becomes a zero.

Usage::

    from sonointent.eval.harness import run_test
    run_test(OUTD, rows, infer_one, "Model-7B", "ckpt_best")

`infer_one(row)` receives one split row (with `_image` / `_mask` absolute paths
added) and returns a dict with `dice, iou, precision, recall, inter, union,
seg_emitted, valid_mask`, plus `generated_text` and any per-sample extras.
"""
import collections
import glob
import json
import os
import sys
import time
import traceback

RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))

FINAL_ROOT = os.environ.get("SONOINTENT_FINAL_ROOT", "")
# `TEST_SPLIT_FILE` redirects `load_rows()` to another split, e.g. the
# target-absent supplementary set.
SPLIT_PATH = os.environ.get(
    "TEST_SPLIT_FILE",
    os.path.join(FINAL_ROOT, "split/test.jsonl") if FINAL_ROOT else "test.jsonl")

MAX_NEW_TOKENS = 384
SUBSETS = ["Overall", "ID", "OOD_dataset", "OOD_intent", "OOD_organ"]

_T0 = time.time()


# --------------------------------------------------------------------- input

def load_rows(split_path=SPLIT_PATH):
    """Read the official test split. `TEST_LIMIT=N` truncates (smoke tests)."""
    rows = [json.loads(l) for l in open(split_path, encoding="utf-8")]
    lim = int(os.environ.get("TEST_LIMIT", "0"))
    if lim > 0:
        rows = rows[:lim]
    if RANK == 0:
        print(f"[lib] {len(rows)} test rows from {split_path}", flush=True)
    for r in rows:
        r["_image"] = os.path.join(FINAL_ROOT, r["rendered_image"])
        r["_mask"] = os.path.join(FINAL_ROOT, r["rendered_mask"])
    return rows


def load_val_rows(split_path=None):
    """Same, for the val split — useful for reproducing training-time numbers."""
    p = split_path or f"{FINAL_ROOT}/split/val.jsonl"
    return load_rows(p)


# ------------------------------------------------------------------ scoring

def score(mask, gt):
    """Official per-sample metrics. `mask` and `gt` are binary tensors/arrays of
    identical shape; `mask` is already thresholded at `pred > 0`.

    Returns (dice, iou, precision, recall, inter, union). An empty prediction
    scores 0 against a non-empty target (that IS a real failure of the model,
    unlike an exception)."""
    import torch
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if not torch.is_tensor(gt):
        gt = torch.as_tensor(gt)
    # Both sides to CPU float: callers hand in a CUDA logits tensor for one side
    # and a numpy GT array for the other.
    mask = mask.detach().cpu().float().reshape(-1)
    gt = gt.detach().cpu().float().reshape(-1)
    sp = float(mask.sum())
    sg = float(gt.sum())
    inter = float((mask * gt).sum())
    union = sp + sg - inter
    dice = 2 * inter / (sp + sg + 1e-7)
    iou = inter / (union + 1e-7)
    prec = inter / (sp + 1e-7)
    rec = inter / (sg + 1e-7)
    return dice, iou, prec, rec, inter, union


# ------------------------------------------------------------------- running

def _base_rec(r):
    # The target-absent supplementary set carries neither key; official test rows
    # always carry both.
    return {
        "sample_id": r["sample_id"], "dataset": r["dataset"],
        "ood_type": r.get("ood_type", "target_absent"), "problem": r["problem"],
        "thinking_gt": r.get("thinking", ""),
        "rendered_image": r["rendered_image"],
        "rendered_mask": r["rendered_mask"],
    }


def run_test(outd, rows, infer_one, model_name, ckpt_name, retry_hook=None):
    """Shard `rows` over WORLD ranks, score each, and publish the report.

    Only rank 0 returns the metrics dict; every other rank returns None.
    Exits 2 (on rank 0) when the run is incomplete.
    """
    os.makedirs(outd, exist_ok=True)
    my_rows = rows[RANK::WORLD]
    rank_pred = f"{outd}/predictions_rank{RANK}.jsonl"
    rank_agg = f"{outd}/agg_rank{RANK}.json"

    # A previous run's leftover rank files must not satisfy this run's wait.
    for stale in glob.glob(f"{outd}/agg_rank*.json") + glob.glob(f"{outd}/predictions_rank*.jsonl"):
        try:
            if os.path.getmtime(stale) < _T0 - 5:
                os.remove(stale)
                print(f"[lib] removed stale {os.path.basename(stale)}", flush=True)
        except OSError:
            pass

    pf = open(rank_pred, "w", encoding="utf-8")
    err_kinds = collections.Counter()
    n_err = 0
    n_scored = 0
    n_retried = 0
    first_tb = []
    start = time.time()

    pending = list(my_rows)
    for attempt in (0, 1):
        failed = []
        for i, r in enumerate(pending):
            try:
                extra = infer_one(r)
                rec = _base_rec(r)
                rec.update(extra)
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pf.flush()
                n_scored += 1
            except Exception as e:
                if attempt == 0:
                    failed.append(r)
                    if len(first_tb) < 3:
                        first_tb.append(traceback.format_exc())
                else:
                    n_err += 1
                    err_kinds[type(e).__name__] += 1
                    if "out of memory" in str(e).lower():
                        err_kinds["CUDA_OOM"] += 1
                    rec = _base_rec(r)
                    rec.update(generated_text=None, dice=None, iou=None,
                               precision=None, recall=None, seg_emitted=False,
                               valid_mask=False, error=str(e)[:200])
                    pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pf.flush()
                    print(f"rank{RANK} SAMPLE_ERROR [{r['dataset']}] "
                          f"{r['sample_id']}: {str(e)[:120]}", flush=True)
            if RANK == 0 and (i + 1) % 200 == 0:
                print(f"rank0 pass{attempt} {i+1}/{len(pending)} "
                      f"({time.time()-start:.0f}s)", flush=True)
        if attempt == 0 and failed:
            n_retried = len(failed)
            print(f"rank{RANK} retrying {len(failed)} failed samples", flush=True)
            for tb in first_tb:
                print("---- first-pass traceback ----\n" + tb, flush=True)
            if retry_hook is not None:
                retry_hook()
            pending = failed
        else:
            break
    pf.close()

    json.dump({"_n_error": n_err, "_error_kinds": dict(err_kinds),
               "_n_scored": n_scored, "_n_expected": len(my_rows),
               "_n_retried": n_retried},
              open(rank_agg, "w", encoding="utf-8"))
    print(f"rank{RANK} done: {n_scored}/{len(my_rows)} scored, {n_err} errored, "
          f"{n_retried} retried, {time.time()-start:.0f}s", flush=True)

    if RANK != 0:
        return None

    # ---- rank 0: wait for every rank, then decide whether to publish --------
    deadline = time.time() + 12 * 3600
    while True:
        ready = [f for f in glob.glob(f"{outd}/agg_rank*.json")
                 if os.path.getmtime(f) >= _T0 - 5]
        if len(ready) >= WORLD:
            break
        if time.time() > deadline:
            print(f"[lib] gave up waiting: {len(ready)}/{WORLD} ranks reported",
                  flush=True)
            sys.exit(2)
        time.sleep(5)

    total_err, total_retried, scored = 0, 0, 0
    kinds = collections.Counter()
    for i in range(WORLD):
        a = json.load(open(f"{outd}/agg_rank{i}.json", encoding="utf-8"))
        total_err += int(a.get("_n_error", 0))
        total_retried += int(a.get("_n_retried", 0))
        scored += int(a.get("_n_scored", 0))
        kinds.update(a.get("_error_kinds", {}))

    preds = []
    for i in range(WORLD):
        # `encoding=` here is load-bearing, not hygiene. Records go out with
        # ensure_ascii=False, so `problem` / `thinking_gt` land as raw UTF-8; if
        # the launching shell's locale is C/POSIX the default codec is 'ascii'
        # and this read dies. That is exactly what happened to the InstructSeg
        # run of 2026-09-25: all 22,870 rows had been scored with 0 errors, and
        # the publish step alone blew up, so metrics.json was never written.
        for l in open(f"{outd}/predictions_rank{i}.jsonl", encoding="utf-8"):
            if l.strip():
                preds.append(json.loads(l))
    preds.sort(key=lambda p: (p["dataset"], p["sample_id"]))

    incomplete = total_err > 0 or len(preds) != len(rows)
    if incomplete:
        print("=" * 72, flush=True)
        print(f"REFUSING TO PUBLISH: {total_err} errored, {scored}/{len(rows)} scored, "
              f"{len(preds)} records for {len(rows)} rows.", flush=True)
        print(f"error kinds: {dict(kinds)}", flush=True)
        print("A missing result is not a zero. Fix the run and re-run; do NOT read "
              "these numbers as performance.", flush=True)
        for stale_name in ("metrics.json", "paper_results.md", "predictions.jsonl"):
            p = f"{outd}/{stale_name}"
            if os.path.exists(p):
                print(f"  STALE (from an earlier run, NOT this one): {p}", flush=True)
        print("=" * 72, flush=True)
        json.dump({"STATUS": "INCOMPLETE_DO_NOT_USE",
                   "n_error": total_err, "error_kinds": dict(kinds),
                   "n_retried": total_retried, "n_scored": scored,
                   "n_expected": len(rows), "n_records": len(preds),
                   "model": model_name, "checkpoint": ckpt_name,
                   "note": ("Samples that raised are excluded from every metric and "
                            "were retried once. No metrics.json / paper_results.md "
                            "was written for this run.")},
                  open(f"{outd}/metrics.INCOMPLETE.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        sys.exit(2)

    with open(f"{outd}/predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    for i in range(WORLD):
        for f in (f"predictions_rank{i}.jsonl", f"agg_rank{i}.json"):
            p = f"{outd}/{f}"
            if os.path.exists(p):
                os.remove(p)

    metrics = build_report(outd, model_name, ckpt_name)
    tot = time.time() - start
    print(f"[lib] PUBLISHED {model_name}: overall Dice "
          f"{metrics['Overall']['dice']:.4f} IoU {metrics['Overall']['iou']:.4f} "
          f"cIoU {metrics['Overall']['ciou']:.4f}; "
          f"{len(preds)} samples, {total_retried} retried, {tot:.0f}s", flush=True)
    return metrics


# ------------------------------------------------------------------ reporting

def build_report(outd, model_name, ckpt_name):
    """Build the canonical metrics tables for a completed run."""
    from .report import build_report as _build
    return _build(outd, model_name, ckpt_name)
