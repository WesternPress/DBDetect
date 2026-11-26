"""Probability-level fusion of finished runs. No training: every weight is chosen on validation AUC and applied to test.

    python -m src.api.fuse models --a retfound_mae_ft_flip --b densenet121_ft_full --name rf+dn
    python -m src.api.fuse eyes   --a retfound_mae_ft_flip --name rf+eyes
    python -m src.api.fuse seeds  --a retfound_mae_ft_flip --name rf_seed_ensemble

models: p = w * p_a + (1 - w) * p_b per seed (seed k of a with seed k of b), w in 0, 0.1, ..., 1.
eyes:   p = (1 - w) * p_eye + w * p_fellow_eye per seed, w in 0, 0.05, ..., 0.5; an image whose other eye is not in the
        same split keeps its own probability.
seeds:  mean probability over the seeds of one arm (one output run).
Run prefixes are relative to paths.runs and may point at earlier fusion outputs (e.g. fusion/rf+dn); a prefix
without per-seed runs (e.g. a seed ensemble) is fused as one run.
Outputs go to runs/fusion/<name>_s<seed>/ (or runs/fusion/<name>/ for seeds): val_preds.csv, test_preds.csv,
fusion.json (sources, weight grid with val AUC, selected weight) and metrics.json from src/api/evaluate.py.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.api.evaluate import evaluate_run, format_summary

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = (0, 1, 50)


def runs_root():
    with open(os.path.join(PROJECT, "config", "base.yaml")) as f:
        return os.path.join(PROJECT, yaml.safe_load(f)["paths"]["runs"])


def load(run_dir, split):
    return pd.read_csv(os.path.join(run_dir, f"{split}_preds.csv"), dtype={"id": str}).set_index("id").sort_index()


def fellow_prob(df):
    """Probability of the other eye of the same patient, or the image's own probability if that eye is absent."""
    other = df.index.str.replace("_left", "_TMP").str.replace("_right", "_left").str.replace("_TMP", "_right")
    return df["prob"].reindex(other).fillna(pd.Series(df["prob"].to_numpy(), index=other)).to_numpy()


def write(out_dir, frames, info):
    os.makedirs(out_dir, exist_ok=True)
    for split, df in frames.items():
        df.reset_index()[["id", "label", "prob", "male"]].to_csv(os.path.join(out_dir, f"{split}_preds.csv"), index=False)
    with open(os.path.join(out_dir, "fusion.json"), "w") as f:
        json.dump(info, f, indent=2)
    return evaluate_run(out_dir)


def pick(grid, combine, val):
    """Weight with the best val AUC; ties go to the smaller weight (closer to the unfused input)."""
    scores = [(float(roc_auc_score(val["label"], combine(w, "val"))), w) for w in grid]
    best = max(scores, key=lambda t: (t[0], -t[1]))
    return best[1], {f"{w:g}": round(s, 5) for s, w in scores}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["models", "eyes", "seeds"])
    p.add_argument("--a", required=True, help="run prefix, e.g. retfound_mae_ft_flip")
    p.add_argument("--b", help="second run prefix (models mode)")
    p.add_argument("--b-seed-offset", type=int, default=0,
                   help="pair seed k of a with seed k+offset of b (e.g. two seeds of one model as a 2-model control)")
    p.add_argument("--name", required=True)
    args = p.parse_args()
    root = runs_root()

    if args.mode == "seeds":
        frames = {}
        for split in ("val", "test"):
            dfs = [load(os.path.join(root, f"{args.a}_s{s}"), split) for s in SEEDS]
            frames[split] = dfs[0].assign(prob=np.mean([d["prob"].to_numpy() for d in dfs], axis=0))
        m = write(os.path.join(root, "fusion", args.name), frames, {"mode": "seeds", "a": args.a, "seeds": SEEDS})
        print(f"{args.name}: {format_summary(m, 'test')}")
        return

    single = not os.path.isdir(os.path.join(root, f"{args.a}_s{SEEDS[0]}"))  # e.g. a seed ensemble
    for s in ([None] if single else SEEDS):
        a_dir = os.path.join(root, args.a if single else f"{args.a}_s{s}")
        A = {sp: load(a_dir, sp) for sp in ("val", "test")}
        if args.mode == "models":
            sb = SEEDS[(SEEDS.index(s) + args.b_seed_offset) % len(SEEDS)]
            B = {sp: load(os.path.join(root, f"{args.b}_s{sb}"), sp) for sp in ("val", "test")}
            assert all((A[sp].index == B[sp].index).all() for sp in A)
            grid = np.round(np.arange(0, 1.0001, 0.1), 2)
            combine = lambda w, sp: w * A[sp]["prob"].to_numpy() + (1 - w) * B[sp]["prob"].to_numpy()
        else:
            F = {sp: fellow_prob(A[sp]) for sp in A}
            grid = np.round(np.arange(0, 0.5001, 0.05), 2)
            combine = lambda w, sp: (1 - w) * A[sp]["prob"].to_numpy() + w * F[sp]
        w, table = pick(grid, combine, A["val"])
        frames = {sp: A[sp].assign(prob=combine(w, sp)) for sp in A}
        info = {"mode": args.mode, "a": args.a, "b": args.b, "seed": s, "weight": float(w), "val_auc_by_weight": table,
                "b_seed_offset": args.b_seed_offset}
        name = args.name if single else f"{args.name}_s{s}"
        m = write(os.path.join(root, "fusion", name), frames, info)
        print(f"{name}: w={w:g} | {format_summary(m, 'test')}")


if __name__ == "__main__":
    main()
