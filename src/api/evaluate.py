"""Evaluation for binary DR detection: overall AUC, sensitivity, specificity, and AUC in female and male groups.

Works on saved prediction CSVs (columns: id, label, prob, male), so metrics can be recomputed without the model:

    python -m src.api.evaluate runs/resnet50_none_s0
    python -m src.api.evaluate runs/resnet50_none_s0 --threshold fixed --fixed-threshold 0.5
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

CI_KEYS = ("auc", "sensitivity", "specificity", "auc_female", "auc_male", "auc_gap", "es_auc")


def _auc(y, prob):
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")


def youden_threshold(y, prob):
    """Threshold that maximizes sensitivity + specificity - 1 (Youden's J)."""
    fpr, tpr, thr = roc_curve(y, prob)
    k = int(np.argmax(tpr - fpr))
    return float(min(thr[k], 1.0))  # roc_curve's first threshold is inf


def compute_metrics(y, prob, male, threshold):
    """Metrics at one decision threshold; a sample is predicted positive when prob >= threshold."""
    y, prob, male = np.asarray(y), np.asarray(prob), np.asarray(male)
    pred = prob >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    auc = _auc(y, prob)
    auc_f = _auc(y[male == 0], prob[male == 0])
    auc_m = _auc(y[male == 1], prob[male == 1])
    return {
        "auc": auc,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "auc_female": auc_f,
        "auc_male": auc_m,
        # Extras, not required by the assignment.
        "accuracy": (tp + tn) / len(y),
        "auc_gap": abs(auc_f - auc_m),
        # Equity-scaled AUC, Luo et al., "FairCLIP", CVPR 2024: AUC / (1 + sum over groups |AUC - AUC_group|).
        "es_auc": auc / (1 + abs(auc - auc_f) + abs(auc - auc_m)),
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_pos": int(np.sum(y == 1)),
    }


def bootstrap_ci(df, threshold, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CIs from resampling patients, so both eyes of a patient are drawn together."""
    patients = df["id"].str.rsplit("_", n=1).str[0].to_numpy()
    rows_of = list(pd.Series(np.arange(len(df))).groupby(patients).apply(np.asarray))
    y, prob, male = df["label"].to_numpy(), df["prob"].to_numpy(), df["male"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in CI_KEYS}
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), len(rows_of))])
        m = compute_metrics(y[idx], prob[idx], male[idx], threshold)
        for k in CI_KEYS:
            samples[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2)))]
            for k, v in samples.items()}


def evaluate_run(run_dir, threshold="youden_val", fixed_threshold=0.5, n_boot=1000, seed=0):
    """Read val_preds.csv / test_preds.csv of a run, pick the threshold, write metrics.json."""
    val = pd.read_csv(os.path.join(run_dir, "val_preds.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(run_dir, "test_preds.csv"), dtype={"id": str})
    if threshold == "youden_val":
        thr = youden_threshold(val["label"], val["prob"])
    elif threshold == "fixed":
        thr = float(fixed_threshold)
    else:
        raise ValueError(f"threshold must be 'youden_val' or 'fixed', got {threshold!r}")

    out = {
        "threshold_mode": threshold,
        "threshold": thr,
        "val": compute_metrics(val["label"], val["prob"], val["male"], thr),
        "test": compute_metrics(test["label"], test["prob"], test["male"], thr),
    }
    if n_boot:
        out["test_ci95"] = bootstrap_ci(test, thr, n_boot, seed)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def format_summary(metrics, split="test"):
    m = metrics[split]
    return (f"{split}: AUC {m['auc']:.4f} | Sens {m['sensitivity']:.4f} | Spec {m['specificity']:.4f} | "
            f"AUC female {m['auc_female']:.4f} | AUC male {m['auc_male']:.4f} | thr {m['threshold']:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--threshold", default="youden_val", choices=["youden_val", "fixed"])
    p.add_argument("--fixed-threshold", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    out = evaluate_run(args.run_dir, args.threshold, args.fixed_threshold, args.bootstrap)
    print(format_summary(out, "val"))
    print(format_summary(out, "test"))


if __name__ == "__main__":
    main()
