"""Penultimate-layer features of a trained run, for fusion heads trained on top of fine-tuned models.

    python -m src.model.run_features --run runs/densenet121_ft_full_s0 --splits val test --gpu 0
    (train_baseline --save-features oof val test does the same at the end of training)

Writes <run>/features_<split>.npz with feature [N, D] float32, id, label, male. "oof" is the held-out cross-fitting fold
of a run trained with data.cv_fold. A feature is the pooled pre-logit representation (timm forward_head with
pre_logits=True), so head(feature) reproduces the model's logit.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from src.model.baselines import build_model
from src.preprocess.dataset import build_dataloaders


@torch.no_grad()
def extract(model, loader, device, amp):
    model.eval()
    feats, ids, labels, male = [], [], [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            f = model.forward_head(model.forward_features(batch["image"].to(device, non_blocking=True)), pre_logits=True)
        feats.append(f.float().cpu().numpy())
        ids += batch["id"]
        labels.append(batch["label"].numpy().astype(int))
        male.append(batch["male"].numpy())
    return {"feature": np.concatenate(feats), "id": np.array(ids), "label": np.concatenate(labels),
            "male": np.concatenate(male)}


def save_features(model, loaders, splits, run_dir, device, amp):
    for split in splits:
        np.savez(os.path.join(run_dir, f"features_{split}.npz"), **extract(model, loaders[split], device, amp))


def holdout_ids(cfg, project):
    """Image ids of the held-out cross-fitting fold of a config, or None."""
    if cfg["data"].get("cv_fold") is None:
        return None
    folds = pd.read_csv(os.path.join(project, cfg["data"]["cv_file"]), dtype={"id": str})
    return set(folds.loc[folds["fold"] == cfg["data"]["cv_fold"], "id"])


def main():
    from src.model.train_baseline import PROJECT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory with config.yaml and best.pt")
    p.add_argument("--splits", nargs="+", default=["val", "test"], help="any of train val test oof")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    with open(os.path.join(args.run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(f"cuda:{args.gpu}")
    model, data_cfg = build_model(cfg["model"], os.path.join(PROJECT, cfg["paths"]["pretrained_cache"]), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(args.run, "best.pt"), map_location="cpu", weights_only=False)["model"])
    model.to(device)
    loaders = build_dataloaders(os.path.join(PROJECT, cfg["data"]["root"]), data_cfg, cfg["data"]["image_size"],
                                cfg["train"]["batch_size"], cfg["data"]["num_workers"], seed=cfg["train"]["seed"],
                                holdout=holdout_ids(cfg, PROJECT))
    save_features(model, loaders, args.splits, args.run, device, cfg["train"]["amp"])
    print(f"{args.run}: wrote features for {args.splits}")


if __name__ == "__main__":
    main()
