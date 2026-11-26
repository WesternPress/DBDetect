"""Feature-level fusion heads on top of fine-tuned backbones (step F3), trained on cross-fitted train features.

    python -m src.model.fusion_head --name rf30+dn_feat --backbones retfound_mae_ft30 densenet121_ft_full
    python -m src.model.fusion_head --name dn_eyes_feat --backbones densenet121_ft_full --eyes
    python -m src.model.fusion_head --name rf30+dn_eyes_feat --backbones retfound_mae_ft30 densenet121_ft_full --eyes --clf mlp

Features per backbone B and seed s:
  train: out-of-fold features, runs/<B>_cv{0..4}_s<s>/features_oof.npz (each fold model never saw its fold's images)
  val/test: the model trained on the whole training set with the same config, runs/<B>_full_s<s> (or runs/<B>_s<s>)
Each block is standardized with train statistics and scaled by 1/sqrt(dim) so blocks weigh equally in the penalty.
--eyes adds, per backbone, the other eye's feature block and |own - other|, plus one flag for a missing other eye (then
the other eye's block repeats the own one). Both eyes of a patient are in the same fold, so the other eye's train
feature is out-of-fold as well.
Ablations: --eye-blocks other / diff keeps only one of the two other-eye blocks; --insample trains the head on the
full-train models' features of their own training images instead of the cross-fitted ones.
Classifier: L2 logistic regression with C chosen on validation AUC (default), or an MLP head (hidden 256, dropout 0.5,
AdamW 1e-3, 100 epochs, best epoch on validation AUC). Outputs runs/fusion/<name>_s<s>/ like src/api/fuse.py.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.api.evaluate import evaluate_run, format_summary
from src.api.fuse import SEEDS, runs_root

C_GRID = 10.0 ** np.arange(-5, 2)


def load_backbone(root, b, seed, insample=False):
    """{split: DataFrame indexed by id with columns label, male and the feature matrix in .attrs['x']}.
    insample=True takes the train features from the full-train model itself instead of the cross-fitted folds
    (the no-cross-fitting ablation: those images were in that model's training set)."""
    full = os.path.join(root, f"{b}_full_s{seed}")
    full = full if os.path.isdir(full) else os.path.join(root, f"{b}_s{seed}")
    if insample:
        tr = [np.load(os.path.join(full, "features_train.npz"))]
    else:
        tr = [np.load(os.path.join(root, f"{b}_cv{k}_s{seed}", "features_oof.npz")) for k in range(5)]
    parts = {"train": tr, "val": [np.load(os.path.join(full, "features_val.npz"))],
             "test": [np.load(os.path.join(full, "features_test.npz"))]}
    out = {}
    for split, arrs in parts.items():
        df = pd.DataFrame({"id": np.concatenate([a["id"] for a in arrs]), "label": np.concatenate([a["label"] for a in arrs]),
                           "male": np.concatenate([a["male"] for a in arrs])})
        x = np.concatenate([a["feature"] for a in arrs])
        order = np.argsort(df["id"].to_numpy())
        df = df.iloc[order].set_index("id")
        df.attrs["x"] = x[order]
        out[split] = df
    assert out["train"].index.is_unique and len(out["train"]) == 4476, "out-of-fold features must cover the train set once"
    return out


def other_eye(index):
    return index.str.replace("_left", "_TMP").str.replace("_right", "_left").str.replace("_TMP", "_right")


def design(blocks, eyes, eye_blocks=("other", "diff")):
    """blocks: list of {split: df} (one per backbone). Returns {split: (X, df)} with standardized, scaled blocks.
    eye_blocks: which other-eye blocks --eyes adds per backbone: the other eye's features and/or |own - other|."""
    ref = blocks[0]
    mats = {s: [] for s in ref}
    for blk in blocks:
        for s in ref:
            assert (blk[s].index == ref[s].index).all()
        mu, sd = blk["train"].attrs["x"].mean(0), blk["train"].attrs["x"].std(0) + 1e-6
        z = {s: (blk[s].attrs["x"] - mu) / sd for s in ref}
        for s in ref:
            pieces = [z[s]]
            if eyes:
                pos = pd.Series(np.arange(len(ref[s])), index=ref[s].index)
                j = pos.reindex(other_eye(ref[s].index)).to_numpy()
                missing = np.isnan(j)
                j = np.where(missing, np.arange(len(j)), j).astype(int)
                pieces += [{"other": z[s][j], "diff": np.abs(z[s] - z[s][j])}[b] for b in eye_blocks]
            mats[s] += [p / np.sqrt(p.shape[1]) for p in pieces]
    out = {}
    for s in ref:
        X = np.concatenate(mats[s], axis=1)
        if eyes:
            pos = pd.Series(np.arange(len(ref[s])), index=ref[s].index)
            X = np.concatenate([X, pos.reindex(other_eye(ref[s].index)).isna().to_numpy()[:, None].astype(float)], axis=1)
        out[s] = (X, ref[s])
    return out


def fit_logreg(D):
    (Xtr, tr), (Xva, va) = D["train"], D["val"]
    scores = []
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=10000).fit(Xtr, tr["label"])
        scores.append((roc_auc_score(va["label"], clf.decision_function(Xva)), -C, clf))
    best = max(scores, key=lambda t: (t[0], t[1]))
    clf = best[2]
    prob = {s: clf.predict_proba(D[s][0])[:, 1] for s in D}
    return prob, {"clf": "logreg", "C": float(-best[1]), "val_auc_by_C": {f"{-c:g}": round(a, 5) for a, c, _ in scores},
                  "train_auc": float(roc_auc_score(tr["label"], prob["train"]))}


def fit_mlp(D, seed, device):
    torch.manual_seed(seed)
    t = {s: torch.tensor(D[s][0], dtype=torch.float32, device=device) for s in D}
    y = torch.tensor(D["train"][1]["label"].to_numpy(), dtype=torch.float32, device=device)
    m = torch.nn.Sequential(torch.nn.Linear(t["train"].shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                            torch.nn.Linear(256, 1)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    epochs, bs, n = 100, 256, len(y)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * ((n + bs - 1) // bs))
    g = torch.Generator().manual_seed(seed)
    best = (-1.0, 0, None)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(t["train"][idx]).squeeze(1), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        m.eval()
        with torch.no_grad():
            v = roc_auc_score(D["val"][1]["label"], m(t["val"]).squeeze(1).cpu().numpy())
        if v > best[0]:
            best = (v, ep, {k: w.clone() for k, w in m.state_dict().items()})
    m.load_state_dict(best[2])
    m.eval()
    with torch.no_grad():
        prob = {s: torch.sigmoid(m(t[s]).squeeze(1)).cpu().numpy() for s in D}
    return prob, {"clf": "mlp", "hidden": 256, "dropout": 0.5, "best_epoch": best[1], "best_val_auc": float(best[0]),
                  "train_auc": float(roc_auc_score(D["train"][1]["label"], prob["train"]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--eyes", action="store_true")
    ap.add_argument("--clf", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--insample", action="store_true",
                    help="ablation: train the head on in-sample train features of the full-train models (no cross-fitting)")
    ap.add_argument("--eye-blocks", choices=["other,diff", "other", "diff"], default="other,diff",
                    help="ablation: other-eye blocks added with --eyes")
    args = ap.parse_args()
    root = runs_root()
    for s in SEEDS:
        # a backbone may be written name:k to take seed s+k of it (e.g. two seeds of one model as a 2-model control)
        specs = [(b.split(":")[0], SEEDS[(SEEDS.index(s) + int(b.split(":")[1])) % len(SEEDS)] if ":" in b else s)
                 for b in args.backbones]
        D = design([load_backbone(root, b, sb, args.insample) for b, sb in specs], args.eyes, tuple(args.eye_blocks.split(",")))
        prob, info = fit_logreg(D) if args.clf == "logreg" else fit_mlp(D, s, torch.device(f"cuda:{args.gpu}"))
        out = os.path.join(root, "fusion", f"{args.name}_s{s}")
        os.makedirs(out, exist_ok=True)
        for split in ("val", "test"):
            df = D[split][1].assign(prob=prob[split]).reset_index()
            df[["id", "label", "prob", "male"]].to_csv(os.path.join(out, f"{split}_preds.csv"), index=False)
        info.update(insample_train_features=args.insample, eye_blocks=args.eye_blocks if args.eyes else None, backbones=args.backbones, backbone_seeds=[sb for _, sb in specs], eyes=args.eyes, seed=s,
                    dim=int(D["train"][0].shape[1]))
        with open(os.path.join(out, "fusion.json"), "w") as f:
            json.dump(info, f, indent=2)
        m = evaluate_run(out)
        print(f"{args.name}_s{s}: train AUC {info['train_auc']:.3f} | {format_summary(m, 'test')}", flush=True)


if __name__ == "__main__":
    main()
