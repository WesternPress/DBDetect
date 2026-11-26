"""Paired, patient-level bootstrap of the test AUC difference between two arms (each arm = mean AUC over its seeds).

    python -m src.api.stats retfound_mae_ft_flip densenet121_ft_full
    python -m src.api.stats fusion/rf+dn retfound_mae_ft_flip

An arm is a run prefix under paths.runs: <prefix>_s{0,1,50} if those exist, otherwise the single run <prefix>.
Both eyes of a patient are resampled together; the same resamples are used for both arms.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.api.fuse import SEEDS, runs_root


def load_arm(prefix, split="test"):
    root = runs_root()
    dirs = [os.path.join(root, f"{prefix}_s{s}") for s in SEEDS]
    if not all(os.path.isdir(d) for d in dirs):
        dirs = [os.path.join(root, prefix)]
    dfs = [pd.read_csv(os.path.join(d, f"{split}_preds.csv"), dtype={"id": str}).set_index("id").sort_index() for d in dirs]
    return dfs[0]["label"].to_numpy(), dfs[0].index, np.stack([d["prob"].to_numpy() for d in dfs])


def paired_bootstrap(a, b, n_boot=2000, seed=0, mask=None):
    """Returns (difference, ci_low, ci_high, two-sided p) of mean AUC(a) - mean AUC(b) on test.
    mask: optional boolean array selecting a subgroup (e.g. female images)."""
    y, ids, pa = load_arm(a)
    yb, idsb, pb = load_arm(b)
    assert (ids == idsb).all() and (y == yb).all()
    if mask is not None:
        y, ids, pa, pb = y[mask], ids[mask], pa[:, mask], pb[:, mask]
    patients = ids.str.rsplit("_", n=1).str[0].to_numpy()
    _, inv = np.unique(patients, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    auc = lambda P, i: np.mean([roc_auc_score(y[i], p[i]) for p in P])
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        d.append(auc(pa, idx) - auc(pb, idx))
    d = np.array(d)
    full = np.arange(len(y))
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return auc(pa, full) - auc(pb, full), *np.percentile(d, [2.5, 97.5]), max(p, 1 / n_boot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    diff, lo, hi, p = paired_bootstrap(args.a, args.b, args.n_boot)
    print(f"{args.a} - {args.b}: AUC diff {diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}")


if __name__ == "__main__":
    main()
