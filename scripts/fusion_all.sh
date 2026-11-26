#!/usr/bin/env bash
# Every fusion experiment in results/summary.md. No training of backbones: needs the finished runs from
# scripts/make_queue.py (groups unified, retfound, crossfit). CPU only, except the two MLP heads (GPU 0).
# Run from the repository root: bash scripts/fusion_all.sh
set -euo pipefail
export PYTHONNOUSERSITE=1

# Probability level (weights chosen on val)
python -m src.api.fuse models --a retfound_mae_ft_flip --b densenet121_ft_full --name rf+dn
python -m src.api.fuse seeds --a retfound_mae_ft_flip --name rf_seedens
python -m src.api.fuse seeds --a densenet121_ft_full --name dn_seedens
python -m src.api.fuse eyes --a retfound_mae_ft_flip --name rf+eyes
python -m src.api.fuse eyes --a densenet121_ft_full --name dn+eyes
python -m src.api.fuse eyes --a resnet50_ft_full --name rn+eyes
python -m src.api.fuse eyes --a vit_b16_ft_full --name vit+eyes
python -m src.api.fuse eyes --a fusion/rf+dn --name rf+dn+eyes
python -m src.api.fuse eyes --a fusion/rf_seedens --name rf_seedens+eyes
python -m src.api.fuse eyes --a fusion/dn_seedens --name dn_seedens+eyes
python -m src.api.fuse models --a retfound_mae_ft30_full --b densenet121_ft_full --name rf30+dn
python -m src.api.fuse eyes --a retfound_mae_ft30_full --name rf30+eyes
python -m src.api.fuse eyes --a fusion/rf30+dn --name rf30+dn+eyes

# Feature level on cross-fitted features
python -m src.model.fusion_head --name rf30_feat --backbones retfound_mae_ft30
python -m src.model.fusion_head --name dn_feat --backbones densenet121_ft_full
python -m src.model.fusion_head --name rf30+dn_feat --backbones retfound_mae_ft30 densenet121_ft_full
python -m src.model.fusion_head --name rf30_eyes_feat --backbones retfound_mae_ft30 --eyes
python -m src.model.fusion_head --name dn_eyes_feat --backbones densenet121_ft_full --eyes
python -m src.model.fusion_head --name rf30+dn_eyes_feat --backbones retfound_mae_ft30 densenet121_ft_full --eyes
python -m src.model.fusion_head --name rf30+dn_eyes_mlp --backbones retfound_mae_ft30 densenet121_ft_full --eyes --clf mlp --gpu 0

# Same-size controls: two seeds of DenseNet instead of RETFound + DenseNet
python -m src.api.fuse models --a densenet121_ft_full --b densenet121_ft_full --b-seed-offset 1 --name dn+dn2
python -m src.api.fuse eyes --a fusion/dn+dn2 --name dn+dn2+eyes
python -m src.model.fusion_head --name dn+dn2_feat --backbones densenet121_ft_full densenet121_ft_full:1
python -m src.model.fusion_head --name dn+dn2_eyes_feat --backbones densenet121_ft_full densenet121_ft_full:1 --eyes
python -m src.model.fusion_head --name dn+dn2_eyes_mlp --backbones densenet121_ft_full densenet121_ft_full:1 --eyes --clf mlp --gpu 0

# Tables and paired tests
python -m src.api.summarize
