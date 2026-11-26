# DBDetect: Diabetic Retinopathy Detection with Binocular Context and Multi-Model Feature Fusion

CSCE566 final project, Problem 1: detect diabetic retinopathy (DR) from 2D colour fundus photographs (ODIR-5K, course
release). Each eye is classified as DR / no DR, and results are reported as overall AUC, sensitivity, specificity, and
AUC in the female and male groups.

**Method.** We fine-tune a retinal foundation model (RETFound, ViT-L/16) and DenseNet-121 and take their penultimate
features. We add the features of the **same patient's other eye** and the absolute difference between the two eyes.
An MLP fusion head is trained on **patient-grouped, cross-fitted (out-of-fold) features**, so it never sees training
features the backbones have memorized.

**Result.** Test AUC **0.841 ± 0.007** (3 seeds), against **0.793** for the strongest single baseline, DenseNet-121
(+0.048, 95% CI [+0.033, +0.062], p < 0.001). Sensitivity, specificity, and female and male AUC all improve.

| Method (test set, mean ± SD over seeds 0/1/50) | AUC | Sensitivity | Specificity | Female AUC | Male AUC |
|---|---|---|---|---|---|
| ResNet-50 (unified fine-tuning recipe) | 0.775 ± 0.009 | 0.590 | 0.794 | 0.768 | 0.781 |
| DenseNet-121 (unified recipe) | 0.793 ± 0.020 | 0.669 | 0.776 | 0.790 | 0.796 |
| ViT-B/16 (unified recipe) | 0.767 ± 0.006 | 0.642 | 0.741 | 0.772 | 0.763 |
| RETFound, frozen + linear head | 0.768 ± 0.001 | 0.662 | 0.740 | 0.763 | 0.771 |
| RETFound, fine-tuned (official recipe) | 0.798 ± 0.003 | 0.720 | 0.715 | 0.791 | 0.805 |
| **Ours: RETFound + DenseNet + other eye, cross-fitted MLP fusion** | **0.841 ± 0.007** | **0.729** | **0.794** | **0.827** | **0.854** |

Main findings (details and all ablations in [`docs/method_and_results.md`](docs/method_and_results.md)):
- **Every part is needed.** Removing the other eye, either backbone or the cross-fitting lowers test AUC by 0.015–0.028
  (each p ≤ 0.001). Without cross-fitting the head fits memorized training features (train AUC 0.99) and falls below
  plain probability averaging. The head type matters little (MLP vs logistic regression: +0.002).
- **The other eye** gives every backbone a significant gain, +0.012 to +0.017.
- **Multi-model fusion** adds about +0.02. With the same number of models, RETFound beats a second DenseNet seed only
  slightly (+0.008 to +0.012, consistent in sign but mostly not significant).
- **The training recipe matters a lot.** The CNNs' paper (from-scratch) recipes give 0.70–0.71, while a common
  fine-tuning recipe gives up to 0.793.
- **RETFound overfits as soon as its learning rate peaks.** LoRA matches full fine-tuning while training 0.26% of the
  parameters.

A step-by-step walkthrough with runnable examples is in
[`notebooks/DBDetect_pipeline.ipynb`](notebooks/DBDetect_pipeline.ipynb). Open directions are in
[`docs/future_work.md`](docs/future_work.md).

## Repository layout

```
DBDetect/
├── config/
│   ├── base.yaml                  # shared settings: data, epochs, evaluation, paths
│   ├── recipes/finetune.yaml      # unified fine-tuning recipe shared by all baselines
│   ├── unified/                   # baselines and RETFound with the unified recipe
│   ├── baselines/                 # each baseline with its paper's original recipe
│   └── pfm/                       # RETFound: official recipe, probes, LoRA, last 4 blocks, LP-FT
├── src/
│   ├── preprocess/
│   │   └── dataset.py             # npz loading (BGR -> RGB), loaders, GPU augmentation, folds
│   ├── model/
│   │   ├── train_baseline.py      # end-to-end training (deterministic, resumable)
│   │   ├── baselines.py           # model builder, optimizers, learning-rate schedules
│   │   ├── pfm.py                 # RETFound loading, LoRA, trainable-parameter selection
│   │   ├── extract_features.py    # frozen RETFound features
│   │   ├── train_linear_probe.py  # linear / MLP heads on frozen features
│   │   ├── run_features.py        # penultimate features of a trained run
│   │   └── fusion_head.py         # feature-level fusion heads (logistic / MLP, other eye)
│   └── api/
│       ├── evaluate.py            # metrics, Youden threshold, patient-level bootstrap CIs
│       ├── fuse.py                # probability-level fusion (models / other eye / seeds)
│       ├── stats.py               # paired patient-level bootstrap
│       └── summarize.py           # all result tables -> results/
├── scripts/
│   ├── dispatch.py                # multi-GPU job queue (memory-aware packing, resumable)
│   ├── make_queue.py              # queue of every training run behind the results
│   └── fusion_all.sh              # every fusion experiment, then the tables
├── splits/
│   └── train_5fold.csv            # patient-grouped, label-stratified folds for cross-fitting
├── results/                       # summary.md and CSV tables (generated)
├── docs/
│   ├── method_and_results.md      # task, protocol, baselines, method, results, ablations
│   └── future_work.md             # open directions
├── notebooks/
│   └── DBDetect_pipeline.ipynb    # step-by-step walkthrough with runnable examples
├── environment.yml                # conda environment
└── LICENSE
```

## Setup

```bash
conda env create -f environment.yml
conda activate dbdetect
```

The environment sets `PYTHONNOUSERSITE=1`, so packages installed under `~/.local` cannot shadow the pinned versions.

**Data** (not included). Put the course release of ODIR-5K at `data/` (a folder or a symlink) with the official split:

```
data/train/*.npz   data/val/*.npz   data/test/*.npz
```

Each file is one eye, named `<patient>_<left|right>.npz`, with keys `slo_fundus` (200×200×3 uint8, **BGR**),
`dr_class` (0/1) and `male` (0/1). `runs/`, `pretrained/`, `features/` and `logs/` are created locally and are not tracked.

**RETFound weights.** The repository [`YukunZhou/RETFound_mae_natureCFP`](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP)
is gated on Hugging Face. Accept its terms, run `huggingface-cli login` once, and the weights download into `pretrained/` on
first use. ImageNet weights for the baselines are downloaded by timm in the same way.

## Usage

All commands run from the repository root.

```bash
# One training run (unified recipe); writes runs/<model>_<augment>_s<seed>/
python -m src.model.train_baseline --config config/unified/densenet121.yaml --gpu 0 --set train.seed=0

# RETFound fine-tuning with its official recipe and flip augmentation
python -m src.model.train_baseline --config config/pfm/retfound_mae_ft.yaml --gpu 0 --set train.seed=0 data.augment=flip

# Resume an interrupted run
python -m src.model.train_baseline --config config/unified/densenet121.yaml --gpu 0 --set train.seed=0 --resume

# Re-score a run's saved predictions
python -m src.api.evaluate runs/densenet121_ft_full_s0
```

### Reproducing every result

```bash
# 1. All end-to-end training runs (105 jobs) through the multi-GPU queue
python scripts/make_queue.py --out logs/queue_all.yaml
python scripts/dispatch.py --queue logs/queue_all.yaml --gpus 0 1 2 3

# 2. Frozen RETFound probes: extract features once, then train the heads
python -m src.model.extract_features --config config/pfm/retfound_mae_lp.yaml --gpu 0
for s in 0 1 50; do
  python -m src.model.train_linear_probe --config config/pfm/retfound_mae_lp.yaml --set train.seed=$s
  python -m src.model.train_linear_probe --config config/pfm/retfound_mae_lp.yaml --run-name retfound_mae_lp_none_lr0.1x_s$s \
      --set train.seed=$s optim.base_lr=5.0e-4
  for h in 256 512; do
    python -m src.model.train_linear_probe --config config/pfm/retfound_mae_mlp.yaml --run-name retfound_mae_mlp_h${h}_s$s \
        --set train.seed=$s model.hidden=$h
  done
done

# 3. Penultimate features of the DenseNet full-train runs (the cross-fitting runs save theirs during training)
for s in 0 1 50; do python -m src.model.run_features --run runs/densenet121_ft_full_s$s --splits train val test; done

# 4. Every fusion experiment, then all tables and paired tests in results/
bash scripts/fusion_all.sh
```

The LP-FT runs (queue group `adapt`) start from the linear-probe heads of step 2, so run step 2 before them, or run
`make_queue.py --only paper,unified,retfound,crossfit` first. Every run is deterministic: same seed, same numbers.

## Evaluation protocol

- Official folders: train 4,476 / val 641 / test 1,914 images.
  - **Note: all 641 val images also appear in test.** Every choice (epoch, recipe, fusion weight, regularization,
    threshold) uses val only; test metrics are reported for every arm.
- Threshold: Youden's J on val, applied unchanged to test.
- Seeds 0, 1 and 50; mean ± SD.
- Comparisons: paired bootstrap over patients (2,000 resamples, both eyes of a patient together).

## References

**Baselines**

1. **ResNet-50**: K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016.
   [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). Weights: torchvision `IMAGENET1K_V2` (timm `resnet50.tv2_in1k`).
2. **DenseNet-121**: G. Huang, Z. Liu, L. van der Maaten, K. Q. Weinberger. *Densely Connected Convolutional Networks*.
   CVPR 2017. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993). Weights: torchvision `IMAGENET1K_V1`
   (timm `densenet121.tv_in1k`).
3. **ViT-B/16**: A. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*.
   ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929). Weights: timm
   `vit_base_patch16_224.augreg2_in21k_ft_in1k`, ImageNet-21k pre-training from A. Steiner et al., *How to Train Your
   ViT? Data, Augmentation, and Regularization in Vision Transformers*, TMLR 2022
   ([arXiv:2106.10270](https://arxiv.org/abs/2106.10270)).

**Foundation model**

4. **RETFound**: Y. Zhou et al. *A foundation model for generalizable disease detection from retinal images*. Nature
   622, 156–163 (2023). [Code](https://github.com/rmaphoh/RETFound); weights
   [`YukunZhou/RETFound_mae_natureCFP`](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP).

**Adaptation methods (RETFound ablations)**

5. **LoRA**: E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
   [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
6. **LP-FT**: A. Kumar, A. Raghunathan, R. Jones, T. Ma, P. Liang. *Fine-Tuning can Distort Pretrained Features and
   Underperform Out-of-Distribution*. ICLR 2022. [arXiv:2202.10054](https://arxiv.org/abs/2202.10054).

**Data and software**

7. **ODIR-5K**: Peking University, Ocular Disease Intelligent Recognition challenge (ODIR-2019); this project uses the
   CSCE566 course release.
8. **timm**: R. Wightman, *PyTorch Image Models*, <https://github.com/huggingface/pytorch-image-models>.
9. **torchvision**: <https://github.com/pytorch/vision>.

## License

MIT, see [`LICENSE`](LICENSE).
