"""Extract frozen-encoder features for every image of the official train/val/test folders.

    python -m src.model.extract_features --config config/pfm/retfound_mae_lp.yaml --gpu 0

Writes features/<features.name>/{train,val,test}.npz (feature [N, D] float32, id, label, male) and meta.json.
The encoder runs in eval mode in float32 with deterministic kernels and the shared preprocessing (resize, normalize),
so the features are exactly what a frozen encoder would see during training without augmentation.
"""
import argparse
import json
import os
import time

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

from src.model.pfm import build_retfound
from src.model.train_baseline import PROJECT, load_config
from src.preprocess.dataset import SPLITS, ODIRDataset, build_transform


@torch.no_grad()
def extract(model, loader, device):
    feats = []
    for batch in loader:
        feats.append(model(batch["image"].to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(feats)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config/pfm/<model>.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", help="output folder name under paths.features (default: features.name)")
    p.add_argument("--limit", type=int, help="random subset of this many images per split (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = os.path.join(PROJECT, cfg["paths"]["features"], args.out or cfg["features"]["name"])
    if os.path.exists(os.path.join(out_dir, "meta.json")) and not args.overwrite:
        raise SystemExit(f"{out_dir} already has features; pass --overwrite to replace them")
    os.makedirs(out_dir, exist_ok=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{args.gpu}")
    model, data_cfg, load_info = build_retfound(cfg["model"], os.path.join(PROJECT, cfg["paths"]["pretrained_cache"]))
    model.to(device)
    transform = build_transform(data_cfg, cfg["data"]["image_size"])

    t0, counts = time.time(), {}
    for split in SPLITS:
        ds = ODIRDataset(os.path.join(PROJECT, cfg["data"]["root"], split), transform, limit=args.limit,
                         seed=cfg["train"]["seed"])
        loader = DataLoader(ds, batch_size=cfg["features"]["batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"], pin_memory=True)
        feats = extract(model, loader, device)
        np.savez(os.path.join(out_dir, f"{split}.npz"), feature=feats, id=np.array(ds.ids), label=ds.labels,
                 male=ds.male)
        counts[split] = int(len(ds))
        print(f"{split}: {feats.shape} | mean {feats.mean():.4f} std {feats.std():.4f} | "
              f"finite {bool(np.isfinite(feats).all())}", flush=True)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "model": cfg["model"], "data_cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in data_cfg.items()},
            "image_size": cfg["data"]["image_size"], "feature_dim": int(feats.shape[1]), "counts": counts,
            "limit": args.limit, "checkpoint_file": os.path.basename(load_info["checkpoint"]),
            "missing_keys": load_info["missing_keys"], "unexpected_keys": load_info["unexpected_keys"],
            "seconds": round(time.time() - t0, 1),
            "versions": {"torch": str(torch.__version__), "timm": str(timm.__version__)},
        }, f, indent=2)
    print(f"wrote {out_dir} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
