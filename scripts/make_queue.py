"""Write the queue of every end-to-end training run behind results/summary.md, for scripts/dispatch.py.

    python scripts/make_queue.py --out logs/queue_all.yaml [--only unified,crossfit]
    python scripts/dispatch.py --queue logs/queue_all.yaml --gpus 0 1 2 3

Run names match the ones src/api/summarize.py reads. Groups:
  paper      each baseline with its paper recipe: augment none/flip/flip_rot at lr x1, plus lr x0.1 for the val-best augment
  unified    ResNet-50, DenseNet-121, ViT-B/16, RETFound and the RETFound linear probe with the unified fine-tuning recipe
  retfound   RETFound with its own recipe: augment none/flip/flip_rot (100 epochs)
  crossfit   5-fold cross-fitting of RETFound (own recipe, flip, 30 epochs) and DenseNet-121 (unified), plus full-train runs
  adapt      RETFound adaptations: LoRA (two lrs), last 4 blocks, LP-FT (needs the linear-probe runs, see README)
mem_gb is the measured peak GPU memory per job (batch 64) used by the dispatcher to pack jobs.
"""
import argparse

import yaml

SEEDS = (0, 1, 50)
CV = lambda k: ["data.cv_file=splits/train_5fold.csv", f"data.cv_fold={k}"]
AUG = {"none": [], "flip": ["data.augment=flip"], "flip_rot": ["data.augment=flip_rot", "data.rotate_deg=15"]}
MEM = {"resnet50": 6, "densenet121": 6, "vit_b16": 8, "retfound": 22}


def job(name, config, sets, mem, feats=None):
    j = {"name": name, "config": config, "set": sets, "mem_gb": mem}
    if feats:
        j["save_features"] = feats
    return j


def groups():
    g = {"paper": [], "unified": [], "retfound": [], "crossfit": [], "adapt": []}
    lr01 = {"resnet50": ("none", "0.01"), "densenet121": ("none", "0.01"), "vit_b16": ("flip_rot", "0.001")}
    for m in ("resnet50", "densenet121", "vit_b16"):
        for s in SEEDS:
            for a, extra in AUG.items():
                g["paper"].append(job(f"{m}_{a}_s{s}", f"config/baselines/{m}.yaml", [f"train.seed={s}"] + extra, MEM[m]))
            a, lr = lr01[m]
            g["paper"].append(job(f"{m}_{a}_lr0.1x_s{s}", f"config/baselines/{m}.yaml",
                                  [f"train.seed={s}", f"optim.base_lr={lr}"] + AUG[a], MEM[m]))
    for s in SEEDS:
        for m, cfg in (("resnet50_ft", "resnet50"), ("densenet121_ft", "densenet121"), ("vit_b16_ft", "vit_b16")):
            g["unified"].append(job(f"{m}_full_s{s}", f"config/unified/{cfg}.yaml", [f"train.seed={s}"], MEM[cfg]))
        g["unified"].append(job(f"retfound_mae_ft_u_full_s{s}", "config/unified/retfound_mae.yaml", [f"train.seed={s}"], 22))
        g["unified"].append(job(f"retfound_mae_lp_u_full_s{s}", "config/unified/retfound_mae_lp.yaml", [f"train.seed={s}"], 10))
        for a, extra in AUG.items():
            g["retfound"].append(job(f"retfound_mae_ft_{a}_s{s}", "config/pfm/retfound_mae_ft.yaml", [f"train.seed={s}"] + extra, 22))
        for k in range(5):
            g["crossfit"].append(job(f"retfound_mae_ft30_cv{k}_s{s}", "config/pfm/retfound_mae_ft_flip30.yaml",
                                     [f"train.seed={s}"] + CV(k), 22, ["oof", "val", "test"]))
            g["crossfit"].append(job(f"densenet121_ft_full_cv{k}_s{s}", "config/unified/densenet121.yaml",
                                     [f"train.seed={s}"] + CV(k), 6, ["oof", "val", "test"]))
        g["crossfit"].append(job(f"retfound_mae_ft30_full_s{s}", "config/pfm/retfound_mae_ft_flip30.yaml", [f"train.seed={s}"], 22,
                                 ["train", "val", "test"]))
        g["adapt"] += [job(f"retfound_mae_lora_lr1e-3_s{s}", "config/pfm/retfound_mae_lora.yaml", [f"train.seed={s}"], 12),
                       job(f"retfound_mae_lora_lr3e-4_s{s}", "config/pfm/retfound_mae_lora.yaml",
                           [f"train.seed={s}", "optim.base_lr=1.2e-3"], 12),
                       job(f"retfound_mae_last4_s{s}", "config/pfm/retfound_mae_last4.yaml", [f"train.seed={s}"], 18),
                       job(f"retfound_mae_lpft_s{s}", "config/pfm/retfound_mae_lpft.yaml", [f"train.seed={s}"], 22)]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated groups (default: all)")
    args = ap.parse_args()
    g = groups()
    names = args.only.split(",") if args.only else list(g)
    jobs = [j for n in names for j in g[n]]
    with open(args.out, "w") as f:
        yaml.safe_dump(jobs, f, sort_keys=False, width=200)
    print(f"{len(jobs)} jobs ({', '.join(f'{n}: {len(g[n])}' for n in names)}) -> {args.out}")


if __name__ == "__main__":
    main()
