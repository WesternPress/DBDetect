"""Linear probe: train a linear head on frozen-encoder features, keep the epoch with the best val AUC, and evaluate it.

    python -m src.model.extract_features --config config/pfm/retfound_mae_lp.yaml --gpu 0     # once
    python -m src.model.train_linear_probe --config config/pfm/retfound_mae_lp.yaml --gpu 0 --set train.seed=1

Same protocol as src/model/train_baseline.py: data order seeded per (seed, epoch), train.epochs epochs, best epoch by val
AUC, last.pt / best.pt / best.json, --resume, threshold from val, metrics from src/api/evaluate.py. Output goes to
runs/<model>_<augment>_s<seed>/. Only augment=none is possible, because the features are fixed.
"""
import argparse
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score

from src.api.evaluate import evaluate_run, format_summary
from src.model.baselines import LRSchedule, build_optimizer
from src.model.train_baseline import PROJECT, epoch_seed, flatten, load_config, save_atomic, seed_everything


def load_features(feat_dir, split, device):
    with np.load(os.path.join(feat_dir, f"{split}.npz")) as d:
        return {"x": torch.from_numpy(d["feature"]).to(device), "y": torch.from_numpy(d["label"]).float().to(device),
                "male": d["male"], "id": [str(i) for i in d["id"]]}


class MLPHead(torch.nn.Sequential):
    """Linear(dim, hidden) - ReLU - Dropout - Linear(hidden, 1), PyTorch default initialization."""

    def __init__(self, dim, hidden, dropout):
        super().__init__(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                         torch.nn.Linear(hidden, 1))


class LinearHead(torch.nn.Linear):
    """Linear layer initialized like the RETFound head: weight ~ trunc_normal(std), bias 0."""

    def __init__(self, dim, std):
        super().__init__(dim, 1)
        torch.nn.init.trunc_normal_(self.weight, std=std)
        torch.nn.init.zeros_(self.bias)


@torch.no_grad()
def predict(head, data, batch_size):
    """Probability for every sample, and the mean (unsmoothed) BCE loss."""
    head.eval()
    logits = torch.cat([head(data["x"][i:i + batch_size]).squeeze(1) for i in range(0, len(data["id"]), batch_size)])
    loss = F.binary_cross_entropy_with_logits(logits, data["y"]).item()
    df = pd.DataFrame({"id": data["id"], "label": data["y"].cpu().numpy().astype(int),
                       "prob": torch.sigmoid(logits).cpu().numpy(), "male": data["male"]})
    return df, loss


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config/pfm/<model>.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override, e.g. train.seed=1")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--run-name", help="default: <model>_<augment>_s<seed>")
    p.add_argument("--features", help="feature folder name under paths.features (default: features.name)")
    p.add_argument("--resume", action="store_true", help="continue the run from its last.pt")
    p.add_argument("--stop-after", type=int, metavar="EPOCH", help="exit after this epoch; continue with --resume")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    cfg["versions"] = {"torch": str(torch.__version__), "timm": str(timm.__version__)}
    tr = cfg["train"]
    if cfg["data"]["augment"] != "none":
        raise SystemExit("a linear probe on fixed features supports only data.augment=none")
    run_name = args.run_name or f"{cfg['model']['name']}_{cfg['data']['augment']}_s{tr['seed']}"
    run_dir = os.path.join(PROJECT, cfg["paths"]["runs"], run_name)
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
    feat_dir = os.path.join(PROJECT, cfg["paths"]["features"], args.features or cfg["features"]["name"])
    device = torch.device(f"cuda:{args.gpu}")

    if args.resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"--resume: no last.pt in {run_dir}")
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        old, new = flatten({k: v for k, v in state["config"].items() if k != "versions"}), \
            flatten({k: v for k, v in cfg.items() if k != "versions"})
        diff = {k: (old.get(k), new.get(k)) for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        if diff:
            raise SystemExit(f"--resume: config differs from the saved run (saved, now): {diff}")
    else:
        if os.path.exists(last_path) or os.path.exists(os.path.join(run_dir, "metrics.json")):
            raise SystemExit(f"{run_dir} already holds a run; use --resume to continue it or --run-name for a new one")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    seed_everything(tr["seed"])
    if tr["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    data = {s: load_features(feat_dir, s, device) for s in ("train", "val", "test")}
    if cfg.get("features", {}).get("standardize"):
        mu, sd = data["train"]["x"].mean(0), data["train"]["x"].std(0) + 1e-6  # train statistics only
        for d in data.values():
            d["x"] = (d["x"] - mu) / sd
    dim = data["train"]["x"].shape[1]
    if cfg["model"].get("head", "linear") == "mlp":
        head = MLPHead(dim, cfg["model"]["hidden"], cfg["model"]["dropout"]).to(device)
    else:
        head = LinearHead(dim, cfg["model"]["head_init_std"]).to(device)
    bs = tr["batch_size"]
    n_train = len(data["train"]["id"])
    optimizer = build_optimizer(head, cfg["optim"], bs)
    schedule = LRSchedule(optimizer, cfg["sched"], tr["epochs"], math.ceil(n_train / bs))
    smoothing = cfg["loss"]["label_smoothing"]
    grad_clip = cfg["optim"].get("grad_clip")
    order_gen = torch.Generator()

    start_epoch, best_auc, best_epoch, log = 1, -1.0, 0, []
    if args.resume:
        head.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state(state["rng"]["cuda"], device)
        start_epoch, best_auc, best_epoch, log = state["epoch"] + 1, state["best_auc"], state["best_epoch"], state["log"]
        print(f"{run_name}: resumed after epoch {state['epoch']} (best so far: epoch {best_epoch}, "
              f"val AUC {best_auc:.4f})", flush=True)
        del state
    print(f"{run_name}: linear probe on {os.path.basename(feat_dir)} features {tuple(data['train']['x'].shape)} | "
          f"lr {optimizer.param_groups[0]['lr']:.3g} | train/val/test {[len(data[s]['id']) for s in data]}", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, tr["epochs"] + 1):
        t0 = time.time()
        head.train()
        perm = torch.randperm(n_train, generator=order_gen.manual_seed(epoch_seed(tr["seed"], epoch, 0)))
        loss_sum = 0.0
        for i in range(0, n_train, bs):
            schedule.before_step()
            idx = perm[i:i + bs].to(device)
            x, y = data["train"]["x"][idx], data["train"]["y"][idx]
            # Label smoothing of a 2-class softmax == BCE on the logit against targets moved towards 0.5.
            loss = F.binary_cross_entropy_with_logits(head(x).squeeze(1), y * (1 - smoothing) + 0.5 * smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()
            loss_sum += loss.item() * len(idx)

        lr = optimizer.param_groups[0]["lr"]
        val, val_loss = predict(head, data["val"], bs)
        val_auc = roc_auc_score(val["label"], val["prob"])
        schedule.end_epoch(epoch, val_auc)
        is_best = val_auc > best_auc
        if is_best:
            best_auc, best_epoch = val_auc, epoch
        log.append({"epoch": epoch, "train_loss": loss_sum / n_train, "val_loss": val_loss, "val_auc": val_auc,
                    "lr": lr, "is_best": is_best, "seconds": time.time() - t0})
        pd.DataFrame(log).to_csv(os.path.join(run_dir, "log.csv"), index=False)

        state = {
            "epoch": epoch, "model": head.state_dict(), "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(), "best_auc": best_auc, "best_epoch": best_epoch, "log": log,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(device)},
            "config": cfg,
        }
        save_atomic(state, last_path)
        if is_best:
            os.link(last_path, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
            with open(os.path.join(run_dir, "best.json"), "w") as f:
                json.dump({
                    "epoch": epoch, "val_auc": val_auc, "val_loss": val_loss, "train_loss": loss_sum / n_train,
                    "lr_this_epoch": lr,
                    "optimizer": {k: v for k, v in optimizer.param_groups[0].items() if k != "params"},
                    "schedule": schedule.summary(), "features": os.path.basename(feat_dir),
                    "model": cfg["model"]["name"], "seed": tr["seed"], "epochs_planned": tr["epochs"],
                    "batch_size": bs, "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "versions": cfg["versions"],
                }, f, indent=2)
        del state
        print(f"epoch {epoch:3d} | loss {loss_sum / n_train:.4f} | val loss {val_loss:.4f} | val AUC {val_auc:.4f}"
              f"{' *' if is_best else '  '} | lr {lr:.3g}", flush=True)
        if args.stop_after and epoch >= args.stop_after and epoch < tr["epochs"]:
            print(f"stopped after epoch {epoch}; continue with --resume", flush=True)
            return

    print(f"best epoch {best_epoch} (val AUC {best_auc:.4f}) | training time this session "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    head.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    for split in ("val", "test"):
        predict(head, data[split], bs)[0].to_csv(os.path.join(run_dir, f"{split}_preds.csv"), index=False)
    ev = cfg["eval"]
    metrics = evaluate_run(run_dir, ev["threshold"], ev["fixed_threshold"], ev["bootstrap"], seed=tr["seed"])
    print(format_summary(metrics, "val"))
    print(format_summary(metrics, "test"), flush=True)


if __name__ == "__main__":
    main()
