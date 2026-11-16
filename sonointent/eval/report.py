# -*- coding: utf-8 -*-
"""Build metrics.json and paper_results.md from a completed predictions.jsonl.

Tables: Overall / ID / OOD_dataset / OOD_intent / OOD_organ, bootstrap
confidence intervals, dataset-level and clinical-target tables, OOD
composition and quality statistics. `target_of()` maps a mask filename to the
canonical target used by both this report and the target-accuracy scorer.
"""
import json, os, re, sys, collections
import numpy as np

SUBSETS = ["Overall", "ID", "OOD_dataset", "OOD_intent", "OOD_organ"]

# target slug (from the mask filename) -> Chinese name (TARGET_EN in full_v1/v1_engine.py)
SLUG_ZH = {
    "fetal_abdominal_cross-section": "胎儿腹围截面",
    "focal_liver_lesion": "肝内占位",
    "left_atrium": "左心房",
    "left_ventricular_cavity": "左心室腔",
    "left_ventricular_myocardium": "左室心肌",
    "common_carotid_intima-media_complex": "颈总动脉内中膜复合体",
    "right_atrium": "右心房",
    "right_ventricle": "右心室",
    "fetal_liver": "胎儿肝",
    "fetal_stomach": "胎儿胃泡",
    "umbilical_vein": "脐静脉",
    "fetal_head": "胎头",
    "pubic_symphysis": "耻骨联合",
    "gi-wall_lesion": "消化道壁占位",
    "prostate_gland": "前列腺腺体",
    "thyroid_gland": "甲状腺腺体",
}
# datasets whose mask filename carries no target slug -> target used by the dataset
DATASET_ZH = {
    "ACOUSLIC-AI": "胎儿腹围截面", "AUL": "肝内占位", "BUID": "乳腺病灶",
    "BUS-BRA": "乳腺病灶", "BUS-UC": "乳腺病灶", "BUS-UCLM": "乳腺病灶",
    "BUSI": "乳腺病灶", "BrEaST": "乳腺病灶", "C-TRUS": "结肠肠壁",
    "CCAUI": "颈总动脉", "DDTI": "甲状腺结节", "Fast-U-Net": "胎儿腹围截面",
    "GIST514-DB": "消化道壁占位", "HC18": "胎头横断面区域", "MMOTU-2D": "卵巢肿块",
    "MMOTU-3D": "卵巢肿块", "MUP + MicroSegNet": "前列腺腺体", "S1": "乳腺病灶",
    "STMUS NDA": "受检肌肉区域", "STU-Hospital": "乳腺病灶", "Segthy": "甲状腺腺体",
    "TG3K": "甲状腺腺体", "TN3K": "甲状腺结节", "regPro (μ-RegPro)": "前列腺腺体",
    "CUBS": "颈总动脉内中膜复合体", "CAMUS": "左心室腔", "CardiacUDC": "左心室腔",
    "EchoNet-Dynamic": "左心室腔", "FASS": "胎儿肝", "FH-PS-AOP (PSFHS)": "胎头",
}


def target_of(p):
    name = p["rendered_mask"].rsplit("/", 1)[1]
    slug = re.sub(r"^.*__", "", name)
    slug = slug.replace("_mask.png", "").replace(".png", "")
    if slug in SLUG_ZH:
        return SLUG_ZH[slug]
    return DATASET_ZH.get(p["dataset"], slug)


def _agg(rows):
    n = len(rows)
    inter = sum(r.get("inter", 0.0) for r in rows)
    union = sum(r.get("union", 0.0) for r in rows)
    return {
        "n": n,
        "dice": float(np.mean([r["dice"] for r in rows])) if n else 0.0,
        "iou": float(np.mean([r["iou"] for r in rows])) if n else 0.0,
        "precision": float(np.mean([r["precision"] for r in rows])) if n else 0.0,
        "recall": float(np.mean([r["recall"] for r in rows])) if n else 0.0,
        "ciou": float(inter / (union + 1e-7)),
        "seg_emission_rate": float(np.mean([float(bool(r["seg_emitted"])) for r in rows])) if n else 0.0,
        "valid_mask_rate": float(np.mean([float(bool(r["valid_mask"])) for r in rows])) if n else 0.0,
    }


def _boot(vals, seed=42, nboot=1000):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=np.float64)
    if len(v) == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, len(v), size=(nboot, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def build_report(outd, model_name="UGround-7B", ckpt_name="ckpt_best (train_v1)"):
    rows = [json.loads(l) for l in open(f"{outd}/predictions.jsonl", encoding="utf-8")]
    for r in rows:
        for k in ("dice", "iou", "precision", "recall"):
            r.setdefault(k, 0.0)
    failed = [r for r in rows if r.get("generated_text") is None or "error" in r]

    metrics = {}
    for k in SUBSETS:
        metrics[k] = {}
    per_ds = collections.defaultdict(list)
    for r in rows:
        per_ds[r["dataset"]].append(r)
    by_sub = collections.defaultdict(list)
    for r in rows:
        by_sub[r["ood_type"]].append(r)
        by_sub["Overall"].append(r)
    for r in rows:
        metrics.setdefault(r["dataset"], {})

    for sub in SUBSETS:
        a = _agg(by_sub[sub])
        metrics[sub] = {"n": a["n"], "dice": a["dice"], "iou": a["iou"], "giou": a["iou"],
                        "ciou": a["ciou"], "seg_emission_rate": a["seg_emission_rate"],
                        "valid_mask_rate": a["valid_mask_rate"],
                        "empty_prediction_rate": 1.0 - a["valid_mask_rate"]}
    for ds, rs in per_ds.items():
        a = _agg(rs)
        metrics[ds] = {"n": a["n"], "dice": a["dice"], "iou": a["iou"], "giou": a["iou"],
                       "ciou": a["ciou"], "seg_emission_rate": a["seg_emission_rate"],
                       "valid_mask_rate": a["valid_mask_rate"],
                       "empty_prediction_rate": 1.0 - a["valid_mask_rate"]}
    json.dump(metrics, open(f"{outd}/metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    L = []
    L.append(f"# {model_name} Test Results")
    L.append("")
    L.append("## 1. Experimental setting")
    L.append("")
    L.append(f"- Model: {model_name}")
    L.append(f"- Checkpoint: {ckpt_name}")
    L.append(f"- Test samples: {len(rows)}")
    for sub in ["ID", "OOD_dataset", "OOD_intent", "OOD_organ"]:
        L.append(f"- {sub}: {metrics[sub]['n']}")
    L.append("- inference: end-to-end, image + problem only")
    L.append("- GT thinking NOT provided")
    L.append("- greedy decoding")
    L.append(f"- failures: {len(failed)}")
    L.append("")
    L.append("## 2. Main performance")
    L.append("")
    L.append("| Subset | N | Dice | IoU | Precision | Recall | SEG emission | Valid mask |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sub in SUBSETS:
        a = _agg(by_sub[sub])
        L.append(f"| {sub} | {a['n']} | {a['dice']:.4f} | {a['iou']:.4f} | {a['precision']:.4f} | "
                 f"{a['recall']:.4f} | {a['seg_emission_rate']:.4f} | {a['valid_mask_rate']:.4f} |")
    L.append("")
    L.append(f"gIoU (mean sample IoU): {metrics['Overall']['giou']:.4f}")
    L.append(f"cIoU (total intersection / total union): {metrics['Overall']['ciou']:.4f}")
    L.append("")
    L.append("## 3. 95% confidence intervals")
    L.append("")
    L.append("Bootstrap n=1000, seed=42, sample-level Dice / IoU.")
    L.append("")
    for sub in SUBSETS:
        rs = by_sub[sub]
        dl = _boot([r["dice"] for r in rs]); il = _boot([r["iou"] for r in rs])
        L.append(f"- {sub}: Dice {np.mean([r['dice'] for r in rs]):.4f} [{dl[0]:.4f}, {dl[1]:.4f}]; "
                 f"IoU {np.mean([r['iou'] for r in rs]):.4f} [{il[0]:.4f}, {il[1]:.4f}]")
        L.append("")
    L.append("## 4. Dataset-level performance")
    L.append("")
    L.append("| Dataset | OOD type | N | Dice | IoU | Precision | Recall | Valid mask rate |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, rs in sorted(per_ds.items(), key=lambda kv: -_agg(kv[1])["dice"]):
        a = _agg(rs)
        L.append(f"| {ds} | {rs[0]['ood_type']} | {a['n']} | {a['dice']:.4f} | {a['iou']:.4f} | "
                 f"{a['precision']:.4f} | {a['recall']:.4f} | {a['valid_mask_rate']:.4f} |")
    L.append("")
    L.append("## 5. Clinical scenario / target performance")
    L.append("")
    L.append("| Scenario / target | OOD type | N | Dice | IoU |")
    L.append("|---|---|---|---|---|")
    sc = collections.defaultdict(list)
    for r in rows:
        sc[(r["dataset"], target_of(r))].append(r)
    for (ds, tg), rs in sorted(sc.items()):
        a = _agg(rs)
        L.append(f"| {ds} / {tg} | {rs[0]['ood_type']} | {a['n']} | {a['dice']:.4f} | {a['iou']:.4f} |")
    L.append("")
    L.append("## 6. OOD composition")
    L.append("")
    for sub in ["OOD_dataset", "OOD_intent", "OOD_organ"]:
        names = sorted({r["dataset"] for r in by_sub[sub]})
        L.append(f"{sub}: " + ", ".join(names))
        for ds in names:
            a = _agg(per_ds[ds])
            L.append(f"- {ds}: N={a['n']}, Dice={a['dice']:.4f}, IoU={a['iou']:.4f}")
        L.append("")
    L.append("## 7. Key observations")
    L.append("")
    L.append(f"- ID Dice = {metrics['ID']['dice']:.4f}")
    L.append(f"- OOD_dataset Dice = {metrics['OOD_dataset']['dice']:.4f}")
    L.append(f"- OOD_organ Dice = {metrics['OOD_organ']['dice']:.4f}")
    L.append(f"- OOD_intent Dice = {metrics['OOD_intent']['dice']:.4f}")
    oi = sorted(per_ds.items(), key=lambda kv: _agg(kv[1])["dice"])
    oi_txt = "，".join(f"{ds} = {_agg(rs)['dice']:.4f}" for ds, rs in oi
                       if rs[0]["ood_type"] == "OOD_intent")
    if oi_txt:
        L.append(f"- OOD_intent by source: {oi_txt}")
    top = sorted(per_ds.items(), key=lambda kv: -_agg(kv[1])["dice"])[:3]
    L.append("- " + " / ".join(f"{ds} ({_agg(rs)['dice']:.4f})" for ds, rs in top)
             + " 等 seen tasks 表现较高")
    n_intent = metrics["OOD_intent"]["n"]
    L.append(f"- Overall Dice {metrics['Overall']['dice']:.4f} 受 OOD_intent"
             f"（{n_intent}/{len(rows)} = {n_intent/len(rows)*100:.1f}%）大量低分样本拖累，不代表 seen 域能力")
    L.append("")
    L.append("## 8. Quality / inference statistics")
    L.append("")
    L.append(f"- total = {len(rows)}")
    L.append(f"- failed = {len(failed)}")
    L.append(f"- SEG emission rate = {metrics['Overall']['seg_emission_rate']:.5f}")
    L.append(f"- valid mask rate = {metrics['Overall']['valid_mask_rate']:.4f}")
    L.append(f"- empty/invalid mask count = {int(round((1-metrics['Overall']['valid_mask_rate'])*len(rows)))}")
    for r in failed:
        L.append(f"- failed sample: {r['sample_id']} | error: {r.get('error', 'generated_text is None')}")
    L.append("")
    open(f"{outd}/paper_results.md", "w", encoding="utf-8").write("\n".join(L))
    print("REPORT written to", f"{outd}/paper_results.md", flush=True)
    print("METRICS " + json.dumps({k: metrics[k] for k in SUBSETS}, ensure_ascii=False), flush=True)
    return metrics


if __name__ == "__main__":
    build_report(sys.argv[1],
                 model_name=(sys.argv[2] if len(sys.argv) > 2 else "UGround-7B"),
                 ckpt_name=(sys.argv[3] if len(sys.argv) > 3 else "ckpt_best (train_v1)"))
