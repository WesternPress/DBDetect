"""Fine-tune one baseline on ODIR-5K, keep the epoch with the best val AUC, and evaluate it on val and test.

Run from the project root in the environment from environment.yml. That environment sets PYTHONNOUSERSITE=1;
in any other environment export it yourself, or packages in ~/.local may shadow the pinned versions (e.g. timm):

    python -m src.model.train_baseline --config config/baselines/resnet50.yaml --gpu 0
    python -m src.model.train_baseline --config config/baselines/vit_b16.yaml --set data.augment=flip train.seed=1
    python -m src.model.train_baseline --config config/baselines/resnet50.yaml --gpu 0 --resume   # continue a run

Writes runs/<model>_<augment>_s<seed>/:
    config.yaml   resolved config          log.csv     one row per epoch
    last.pt       full training state after the latest epoch (used by --resume)
    best.pt       full training state after the epoch with the best val AUC
    best.json     readable summary of that epoch (metrics, lr, optimizer and schedule state)
    val_preds.csv, test_preds.csv, metrics.json   predictions and metrics of best.pt
"""
import argparse
import json
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
from src.model.baselines import LRSchedule, build_model, build_optimizer
from src.model.run_features import holdout_ids, save_features
from src.preprocess.dataset import (AUGMENTS, build_dataloaders, color_jitter, random_hflip,
                                    random_resized_crop, random_rotate)

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path, overrides=()):
    """config/base.yaml, then the recipe file the model file names under `recipe` (if any), then the model file,
    then `section.key=value` overrides (values parsed as YAML)."""
    with open(os.path.join(PROJECT, "config", "base.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(path) as f:
        model_cfg = yaml.safe_load(f)
    if "recipe" in model_cfg:
        with open(os.path.join(PROJECT, model_cfg["recipe"])) as f:
            deep_update(cfg, yaml.safe_load(f))
    deep_update(cfg, model_cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, last = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return cfg


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epoch_seed(seed, epoch, stream):
    """Seed for one random stream (0: data order, 1: flips, 2: rotations, 3: crops, 4: color jitter) of one epoch, independent of earlier epochs."""
    return (seed * 1000 + epoch) * 10 + stream


def save_atomic(obj, path):
    """Write to a temporary file first, so an interrupted save never leaves a broken checkpoint."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def predict(model, loader, device, amp):
    """Probability for every image of a loader, and the mean BCE loss."""
    model.eval()
    ids, labels, probs, male = [], [], [], []
    loss_sum = 0.0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(batch["image"].to(device, non_blocking=True)).squeeze(1)
        logits = logits.float()
        loss_sum += F.binary_cross_entropy_with_logits(logits, batch["label"].to(device), reduction="sum").item()
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["label"].numpy().astype(int))
        male.append(batch["male"].numpy())
        ids += batch["id"]
    df = pd.DataFrame({"id": ids, "label": np.concatenate(labels), "prob": np.concatenate(probs),
                       "male": np.concatenate(male)})
    return df, loss_sum / len(df)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config/baselines/<model>.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override, e.g. train.seed=1")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--run-name", help="default: <model>_<augment>_s<seed>")
    p.add_argument("--resume", action="store_true", help="continue the run from its last.pt")
    p.add_argument("--stop-after", type=int, metavar="EPOCH", help="exit after this epoch; continue with --resume")
    p.add_argument("--limit", type=int, help="random subset of this many images per split (smoke test)")
    p.add_argument("--save-features", nargs="*", default=[], metavar="SPLIT",
                   help="after training, save penultimate features of the best epoch for these splits (train val test oof)")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    cfg["versions"] = {"torch": str(torch.__version__), "timm": str(timm.__version__)}
    tr = cfg["train"]
    augment = cfg["data"]["augment"]
    if augment not in AUGMENTS:
        raise SystemExit(f"data.augment must be one of {AUGMENTS}, got {augment!r}")
    run_name = args.run_name or f"{cfg['model']['name']}_{augment}_s{tr['seed']}"
    run_dir = os.path.join(PROJECT, cfg["paths"]["runs"], run_name)
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
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
        # Same seed -> same numbers, and --resume continues exactly; costs some convolution speed.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    model, data_cfg = build_model(cfg["model"], os.path.join(PROJECT, cfg["paths"]["pretrained_cache"]),
                                  pretrained=not args.resume)
    if cfg["model"].get("init_head_from") and not args.resume:
        # LP-FT (Kumar et al., 2022): start the head from a trained linear probe of the same seed.
        head_path = os.path.join(PROJECT, cfg["model"]["init_head_from"].format(seed=tr["seed"]))
        model.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=False)["model"])
        print(f"head initialized from {cfg['model']['init_head_from'].format(seed=tr['seed'])}", flush=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_train:,} of {sum(p.numel() for p in model.parameters()):,}", flush=True)
    model.to(device)
    loaders = build_dataloaders(os.path.join(PROJECT, cfg["data"]["root"]), data_cfg, cfg["data"]["image_size"],
                                tr["batch_size"], cfg["data"]["num_workers"], limit=args.limit, seed=tr["seed"],
                                holdout=holdout_ids(cfg, PROJECT))  # cross-fitting: data.cv_file + data.cv_fold
    optimizer = build_optimizer(model, cfg["optim"], tr["batch_size"])
    schedule = LRSchedule(optimizer, cfg["sched"], tr["epochs"], len(loaders["train"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    smoothing = cfg.get("loss", {}).get("label_smoothing", 0.0)
    grad_clip = cfg["optim"].get("grad_clip")
    flip_gen, rot_gen, crop_gen, jit_gen = (torch.Generator(device=device) for _ in range(4))
    if augment in ("flip_rot", "full"):
        rotate_deg = cfg["data"]["rotate_deg"]  # set with --set data.rotate_deg=15 or in a recipe
        black = (-torch.tensor(data_cfg["mean"]) / torch.tensor(data_cfg["std"])).float().to(device)
    if augment == "full":
        mean_t = torch.tensor(data_cfg["mean"]).float().view(1, -1, 1, 1).to(device)
        std_t = torch.tensor(data_cfg["std"]).float().view(1, -1, 1, 1).to(device)

    start_epoch, best_auc, best_epoch, log = 1, -1.0, 0, []
    if args.resume:
        model.load_state_dict(state["model"])
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
    model_id = cfg["model"].get("timm_name") or cfg["model"]["hf_repo"]
    print(f"{run_name}: {model_id} | lr {cfg['optim']['base_lr'] * tr['batch_size'] / cfg['optim']['base_batch_size']:.3g} | "
          f"augment {augment} | mean {data_cfg['mean']} std {data_cfg['std']} {data_cfg['interpolation']} | "
          f"train/val/test{'/oof' if 'oof' in loaders else ''} {[len(l.dataset) for l in loaders.values()]}", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, tr["epochs"] + 1):
        t0 = time.time()
        loaders["train"].sampler.generator.manual_seed(epoch_seed(tr["seed"], epoch, 0))
        flip_gen.manual_seed(epoch_seed(tr["seed"], epoch, 1))
        rot_gen.manual_seed(epoch_seed(tr["seed"], epoch, 2))
        crop_gen.manual_seed(epoch_seed(tr["seed"], epoch, 3))
        jit_gen.manual_seed(epoch_seed(tr["seed"], epoch, 4))
        model.train()
        loss_sum, n = 0.0, 0
        for batch in loaders["train"]:
            schedule.before_step()
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            if augment == "full":
                x = random_resized_crop(x, crop_gen, cfg["data"]["crop_scale"], cfg["data"]["crop_ratio"])
            if augment in ("flip", "flip_rot", "full"):
                x = random_hflip(x, flip_gen)
            if augment in ("flip_rot", "full"):
                x = random_rotate(x, rot_gen, rotate_deg, black)
            if augment == "full":
                x = color_jitter(x, jit_gen, cfg["data"]["jitter"], mean_t, std_t)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tr["amp"]):
                logits = model(x).squeeze(1)
            if smoothing:
                # Label smoothing of a 2-class softmax == BCE on the logit against targets moved towards 0.5.
                y = y * (1 - smoothing) + 0.5 * smoothing
            loss = loss_fn(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += loss.item() * len(y)
            n += len(y)

        top_group = max(optimizer.param_groups, key=lambda g: g.get("lr_scale", 1.0))  # head lr under layer decay
        lr = top_group["lr"]
        val, val_loss = predict(model, loaders["val"], device, tr["amp"])
        val_auc = roc_auc_score(val["label"], val["prob"])
        schedule.end_epoch(epoch, val_auc)
        is_best = val_auc > best_auc
        if is_best:
            best_auc, best_epoch = val_auc, epoch
        log.append({"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss, "val_auc": val_auc, "lr": lr,
                    "is_best": is_best, "seconds": time.time() - t0})
        pd.DataFrame(log).to_csv(os.path.join(run_dir, "log.csv"), index=False)

        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(), "best_auc": best_auc, "best_epoch": best_epoch, "log": log,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(device)},
            "config": cfg,
        }
        # last.pt every save_last_every epochs (plus best epochs, the final epoch and --stop-after); a resume repeats the
        # unsaved epochs exactly, since training is deterministic.
        if (is_best or epoch % tr.get("save_last_every", 1) == 0 or epoch == tr["epochs"]
                or (args.stop_after and epoch >= args.stop_after)):
            save_atomic(state, last_path)
        if is_best:
            # Hard link instead of a second write: last.pt is later replaced by a new file, so best.pt keeps this one.
            os.link(last_path, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
            with open(os.path.join(run_dir, "best.json"), "w") as f:
                json.dump({
                    "epoch": epoch, "val_auc": val_auc, "val_loss": val_loss, "train_loss": loss_sum / n,
                    "lr_this_epoch": lr,
                    "optimizer": {k: v for k, v in top_group.items() if k != "params"},
                    "schedule": schedule.summary(),
                    "model": model_id, "augment": augment, "seed": tr["seed"],
                    "epochs_planned": tr["epochs"], "batch_size": tr["batch_size"],
                    "saved_at": datetime.now().isoformat(timespec="seconds"), "versions": cfg["versions"],
                }, f, indent=2)
        del state
        print(f"epoch {epoch:3d} | loss {loss_sum / n:.4f} | val loss {val_loss:.4f} | val AUC {val_auc:.4f}"
              f"{' *' if is_best else '  '} | lr {lr:.3g} | {time.time() - t0:.0f}s", flush=True)
        if args.stop_after and epoch >= args.stop_after and epoch < tr["epochs"]:
            print(f"stopped after epoch {epoch}; continue with --resume", flush=True)
            return

    print(f"best epoch {best_epoch} (val AUC {best_auc:.4f}) | training time this session "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    for split in ("val", "test") + (("oof",) if "oof" in loaders else ()):
        predict(model, loaders[split], device, tr["amp"])[0].to_csv(os.path.join(run_dir, f"{split}_preds.csv"),
                                                                    index=False)
    if args.save_features:
        save_features(model, loaders, args.save_features, run_dir, device, tr["amp"])
    ev = cfg["eval"]
    metrics = evaluate_run(run_dir, ev["threshold"], ev["fixed_threshold"], ev["bootstrap"], seed=tr["seed"])
    print(format_summary(metrics, "val"))
    print(format_summary(metrics, "test"), flush=True)


if __name__ == "__main__":
    main()
