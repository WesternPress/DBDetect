"""Collect the metrics of finished runs into tables, plus paired bootstrap comparisons.

    python -m src.api.summarize            # writes results/summary.md and results/<table>.csv

An arm is a run prefix under paths.runs (seeds 0/1/50 averaged: mean ± SD) or a single run. Arms that are not finished
yet are listed as pending. Validation AUC is what every selection used; test metrics are reported for every arm.
"""
import json
import os
import statistics as st

import pandas as pd

from src.api.fuse import PROJECT, SEEDS, runs_root
from src.api.stats import paired_bootstrap

TABLES = {
    "main": [
        ("ResNet-50, unified recipe", "resnet50_ft_full"),
        ("DenseNet-121, unified recipe", "densenet121_ft_full"),
        ("ViT-B/16, unified recipe", "vit_b16_ft_full"),
        ("RETFound frozen + linear head", "retfound_mae_lp_none"),
        ("RETFound frozen + MLP head", "retfound_mae_mlp_h256"),
        ("RETFound fine-tuned, own recipe + flip", "retfound_mae_ft_flip"),
    ],
    "fusion": [
        ("RETFound + DenseNet (prob.)", "fusion/rf+dn"),
        ("RETFound 3-seed ensemble", "fusion/rf_seedens"),
        ("DenseNet 3-seed ensemble", "fusion/dn_seedens"),
        ("RETFound + eyes (prob.)", "fusion/rf+eyes"),
        ("DenseNet + eyes (prob.)", "fusion/dn+eyes"),
        ("ResNet + eyes (prob.)", "fusion/rn+eyes"),
        ("ViT-B + eyes (prob.)", "fusion/vit+eyes"),
        ("RETFound + DenseNet + eyes (prob.)", "fusion/rf+dn+eyes"),
        ("RETFound ens. + eyes (prob.)", "fusion/rf_seedens+eyes"),
        ("DenseNet ens. + eyes (prob.)", "fusion/dn_seedens+eyes"),
        ("RETFound30 + DenseNet (prob.)", "fusion/rf30+dn"),
        ("RETFound30 + DenseNet + eyes (prob.)", "fusion/rf30+dn+eyes"),
        ("RETFound30 features, logreg", "fusion/rf30_feat"),
        ("DenseNet features, logreg", "fusion/dn_feat"),
        ("RETFound30 + DenseNet features, logreg", "fusion/rf30+dn_feat"),
        ("RETFound30 + eyes features, logreg", "fusion/rf30_eyes_feat"),
        ("DenseNet + eyes features, logreg", "fusion/dn_eyes_feat"),
        ("RETFound30 + DenseNet + eyes features, logreg", "fusion/rf30+dn_eyes_feat"),
        ("RETFound30 + DenseNet + eyes features, MLP", "fusion/rf30+dn_eyes_mlp"),
        ("control: DenseNet + DenseNet (2 seeds, prob.)", "fusion/dn+dn2"),
        ("control: DenseNet + DenseNet + eyes (prob.)", "fusion/dn+dn2+eyes"),
        ("control: DenseNet + DenseNet features, logreg", "fusion/dn+dn2_feat"),
        ("control: DenseNet + DenseNet + eyes features, logreg", "fusion/dn+dn2_eyes_feat"),
        ("control: DenseNet + DenseNet + eyes features, MLP", "fusion/dn+dn2_eyes_mlp"),
    ],
    "adaptation": [
        ("frozen + linear (RETFound recipe)", "retfound_mae_lp_none"),
        ("frozen + linear (unified recipe, aug.)", "retfound_mae_lp_u_full"),
        ("frozen + MLP 256", "retfound_mae_mlp_h256"),
        ("frozen + MLP 512", "retfound_mae_mlp_h512"),
        ("LoRA r8 on qkv, lr 1e-3", "retfound_mae_lora_lr1e-3"),
        ("LoRA r8 on qkv, lr 3e-4", "retfound_mae_lora_lr3e-4"),
        ("last 4 blocks", "retfound_mae_last4"),
        ("LP-FT", "retfound_mae_lpft"),
        ("full fine-tuning, 30 epochs", "retfound_mae_ft30_full"),
        ("full fine-tuning, 100 epochs", "retfound_mae_ft_flip"),
        ("full fine-tuning, unified recipe", "retfound_mae_ft_u_full"),
    ],
    "recipes": [
        ("ResNet-50 paper x1 none", "resnet50_none"), ("ResNet-50 paper x1 flip", "resnet50_flip"),
        ("ResNet-50 paper x1 flip_rot", "resnet50_flip_rot"), ("ResNet-50 paper x0.1 none", "resnet50_none_lr0.1x"),
        ("DenseNet-121 paper x1 none", "densenet121_none"), ("DenseNet-121 paper x1 flip", "densenet121_flip"),
        ("DenseNet-121 paper x1 flip_rot", "densenet121_flip_rot"), ("DenseNet-121 paper x0.1 none", "densenet121_none_lr0.1x"),
        ("ViT-B/16 paper x1 none", "vit_b16_none"), ("ViT-B/16 paper x1 flip", "vit_b16_flip"),
        ("ViT-B/16 paper x1 flip_rot", "vit_b16_flip_rot"), ("ViT-B/16 paper x0.1 flip_rot", "vit_b16_flip_rot_lr0.1x"),
        ("RETFound own x1 none", "retfound_mae_ft_none"), ("RETFound own x1 flip", "retfound_mae_ft_flip"),
        ("RETFound own x1 flip_rot", "retfound_mae_ft_flip_rot"), ("RETFound frozen + linear x0.1", "retfound_mae_lp_none_lr0.1x"),
    ],
}

COMPARISONS = [
    ("retfound_mae_ft_flip", "densenet121_ft_full"), ("retfound_mae_ft_flip", "vit_b16_ft_full"),
    ("fusion/rf+dn", "fusion/dn_seedens"), ("fusion/rf+eyes", "retfound_mae_ft_flip"), ("fusion/dn+eyes", "densenet121_ft_full"),
    ("fusion/rf+dn+eyes", "densenet121_ft_full"), ("fusion/rf+dn+eyes", "fusion/dn_seedens+eyes"),
    ("fusion/rf30+dn_feat", "fusion/rf30+dn"), ("fusion/rf30+dn_eyes_feat", "fusion/rf30+dn+eyes"),
    ("fusion/rf30+dn_eyes_feat", "densenet121_ft_full"), ("fusion/dn_eyes_feat", "fusion/dn+eyes"),
    ("retfound_mae_lora_lr1e-3", "retfound_mae_ft30_full"), ("retfound_mae_last4", "retfound_mae_ft30_full"),
    ("retfound_mae_lpft", "retfound_mae_ft30_full"), ("retfound_mae_lora_lr3e-4", "retfound_mae_ft30_full"),
    ("retfound_mae_mlp_h256", "retfound_mae_lp_none"),
    # 2-model controls: is RETFound better than a second DenseNet seed?
    ("fusion/rf30+dn", "fusion/dn+dn2"), ("fusion/rf30+dn+eyes", "fusion/dn+dn2+eyes"),
    ("fusion/rf30+dn_feat", "fusion/dn+dn2_feat"), ("fusion/rf30+dn_eyes_feat", "fusion/dn+dn2_eyes_feat"),
    ("fusion/rf30+dn_eyes_mlp", "fusion/dn+dn2_eyes_mlp"), ("fusion/rf30+dn_eyes_mlp", "fusion/rf30+dn_eyes_feat"),
    ("fusion/rf30+dn_eyes_mlp", "densenet121_ft_full"), ("fusion/rf30+dn_eyes_mlp", "retfound_mae_ft_flip"),
]


def run_dirs(prefix):
    root = runs_root()
    dirs = [os.path.join(root, f"{prefix}_s{s}") for s in SEEDS]
    if all(os.path.exists(os.path.join(d, "metrics.json")) for d in dirs):
        return dirs
    single = os.path.join(root, prefix)
    return [single] if os.path.exists(os.path.join(single, "metrics.json")) else None


def row(label, prefix):
    dirs = run_dirs(prefix)
    if dirs is None:
        return {"arm": label, "run": prefix, "n": 0}
    ms = [json.load(open(os.path.join(d, "metrics.json"))) for d in dirs]
    out = {"arm": label, "run": prefix, "n": len(ms)}
    for col, part, key in [("val AUC", "val", "auc"), ("test AUC", "test", "auc"), ("Sens", "test", "sensitivity"),
                           ("Spec", "test", "specificity"), ("AUC female", "test", "auc_female"), ("AUC male", "test", "auc_male")]:
        v = [m[part][key] for m in ms]
        out[col] = st.mean(v)
        out[col + " sd"] = st.stdev(v) if len(v) > 1 else float("nan")
    out["F - M"] = st.mean(m["test"]["auc_female"] - m["test"]["auc_male"] for m in ms)
    return out


def fmt(r):
    if r["n"] == 0:
        return f"| {r['arm']} | `{r['run']}` | pending | | | | | | | |"
    pm = lambda c: f"{r[c]:.3f}" + ("" if r["n"] == 1 else f" ± {r[c + ' sd']:.3f}")
    return (f"| {r['arm']} | `{r['run']}` | {r['n']} | {r['val AUC']:.3f} | {pm('test AUC')} | {r['Sens']:.3f} | "
            f"{r['Spec']:.3f} | {r['AUC female']:.3f} | {r['AUC male']:.3f} | {r['F - M']:+.3f} |")


def main():
    out_dir = os.path.join(PROJECT, "results")
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# Results", "", "Test set: official test folder (1,914 images). Seeds 0/1/50 (mean ± SD) unless n = 1. "
             "Every selection (epoch, recipe, fusion weight, C) used validation AUC only.", ""]
    for name, arms in TABLES.items():
        rows = [row(l, p) for l, p in arms]
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        lines += [f"## {name}", "", "| arm | run | n | val AUC | test AUC | Sens | Spec | AUC female | AUC male | F - M |",
                  "|---|---|---|---|---|---|---|---|---|---|"] + [fmt(r) for r in rows] + [""]
    lines += ["## paired bootstrap (test AUC, patient-level, 2000 resamples)", "", "| A | B | A - B | 95% CI | p |", "|---|---|---|---|---|"]
    comp = []
    for a, b in COMPARISONS:
        if run_dirs(a) is None or run_dirs(b) is None:
            lines.append(f"| `{a}` | `{b}` | pending | | |")
            continue
        d, lo, hi, p = paired_bootstrap(a, b)
        comp.append({"A": a, "B": b, "diff": d, "ci_low": lo, "ci_high": hi, "p": p})
        lines.append(f"| `{a}` | `{b}` | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} |")
    pd.DataFrame(comp).to_csv(os.path.join(out_dir, "comparisons.csv"), index=False)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
