# Results

Test set: official test folder (1,914 images). Seeds 0/1/50 (mean ± SD) unless n = 1. Every selection (epoch, recipe, fusion weight, C) used validation AUC only.

## main

| arm | run | n | val AUC | test AUC | Sens | Spec | AUC female | AUC male | F - M |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50, unified recipe | `resnet50_ft_full` | 3 | 0.796 | 0.775 ± 0.009 | 0.590 | 0.794 | 0.768 | 0.781 | -0.013 |
| DenseNet-121, unified recipe | `densenet121_ft_full` | 3 | 0.802 | 0.793 ± 0.020 | 0.669 | 0.776 | 0.790 | 0.796 | -0.006 |
| ViT-B/16, unified recipe | `vit_b16_ft_full` | 3 | 0.772 | 0.767 ± 0.006 | 0.642 | 0.741 | 0.772 | 0.763 | +0.009 |
| RETFound frozen + linear head | `retfound_mae_lp_none` | 3 | 0.770 | 0.768 ± 0.001 | 0.662 | 0.740 | 0.763 | 0.771 | -0.009 |
| RETFound frozen + MLP head | `retfound_mae_mlp_h256` | 3 | 0.793 | 0.780 ± 0.004 | 0.683 | 0.723 | 0.774 | 0.786 | -0.013 |
| RETFound fine-tuned, own recipe + flip | `retfound_mae_ft_flip` | 3 | 0.805 | 0.798 ± 0.003 | 0.720 | 0.715 | 0.791 | 0.805 | -0.015 |

## fusion

| arm | run | n | val AUC | test AUC | Sens | Spec | AUC female | AUC male | F - M |
|---|---|---|---|---|---|---|---|---|---|
| RETFound + DenseNet (prob.) | `fusion/rf+dn` | 3 | 0.829 | 0.819 ± 0.012 | 0.689 | 0.782 | 0.811 | 0.826 | -0.015 |
| RETFound 3-seed ensemble | `fusion/rf_seedens` | 1 | 0.816 | 0.809 | 0.700 | 0.757 | 0.800 | 0.816 | -0.016 |
| DenseNet 3-seed ensemble | `fusion/dn_seedens` | 1 | 0.829 | 0.818 | 0.757 | 0.726 | 0.817 | 0.819 | -0.003 |
| RETFound + eyes (prob.) | `fusion/rf+eyes` | 3 | 0.814 | 0.812 ± 0.005 | 0.741 | 0.732 | 0.801 | 0.822 | -0.021 |
| DenseNet + eyes (prob.) | `fusion/dn+eyes` | 3 | 0.819 | 0.810 ± 0.016 | 0.697 | 0.770 | 0.809 | 0.811 | -0.002 |
| ResNet + eyes (prob.) | `fusion/rn+eyes` | 3 | 0.814 | 0.790 ± 0.007 | 0.662 | 0.766 | 0.784 | 0.796 | -0.013 |
| ViT-B + eyes (prob.) | `fusion/vit+eyes` | 3 | 0.783 | 0.779 ± 0.006 | 0.678 | 0.730 | 0.779 | 0.779 | -0.000 |
| RETFound + DenseNet + eyes (prob.) | `fusion/rf+dn+eyes` | 3 | 0.842 | 0.836 ± 0.010 | 0.759 | 0.759 | 0.826 | 0.845 | -0.019 |
| RETFound ens. + eyes (prob.) | `fusion/rf_seedens+eyes` | 1 | 0.826 | 0.824 | 0.747 | 0.750 | 0.813 | 0.834 | -0.021 |
| DenseNet ens. + eyes (prob.) | `fusion/dn_seedens+eyes` | 1 | 0.843 | 0.832 | 0.694 | 0.820 | 0.830 | 0.834 | -0.004 |
| RETFound30 + DenseNet (prob.) | `fusion/rf30+dn` | 3 | 0.829 | 0.819 ± 0.012 | 0.689 | 0.782 | 0.811 | 0.826 | -0.015 |
| RETFound30 + DenseNet + eyes (prob.) | `fusion/rf30+dn+eyes` | 3 | 0.842 | 0.836 ± 0.010 | 0.759 | 0.759 | 0.826 | 0.845 | -0.019 |
| RETFound30 features, logreg | `fusion/rf30_feat` | 3 | 0.805 | 0.799 ± 0.002 | 0.719 | 0.715 | 0.790 | 0.806 | -0.015 |
| DenseNet features, logreg | `fusion/dn_feat` | 3 | 0.813 | 0.803 ± 0.004 | 0.649 | 0.807 | 0.798 | 0.807 | -0.010 |
| RETFound30 + DenseNet features, logreg | `fusion/rf30+dn_feat` | 3 | 0.827 | 0.824 ± 0.005 | 0.731 | 0.754 | 0.815 | 0.831 | -0.016 |
| RETFound30 + eyes features, logreg | `fusion/rf30_eyes_feat` | 3 | 0.811 | 0.815 ± 0.003 | 0.736 | 0.743 | 0.801 | 0.827 | -0.027 |
| DenseNet + eyes features, logreg | `fusion/dn_eyes_feat` | 3 | 0.823 | 0.818 ± 0.006 | 0.673 | 0.813 | 0.814 | 0.823 | -0.009 |
| RETFound30 + DenseNet + eyes features, logreg | `fusion/rf30+dn_eyes_feat` | 3 | 0.839 | 0.838 ± 0.003 | 0.733 | 0.783 | 0.829 | 0.848 | -0.019 |
| RETFound30 + DenseNet + eyes features, MLP | `fusion/rf30+dn_eyes_mlp` | 3 | 0.844 | 0.841 ± 0.007 | 0.729 | 0.794 | 0.827 | 0.854 | -0.027 |
| control: DenseNet + DenseNet (2 seeds, prob.) | `fusion/dn+dn2` | 3 | 0.823 | 0.810 ± 0.012 | 0.733 | 0.734 | 0.808 | 0.812 | -0.003 |
| control: DenseNet + DenseNet + eyes (prob.) | `fusion/dn+dn2+eyes` | 3 | 0.838 | 0.824 ± 0.012 | 0.706 | 0.790 | 0.824 | 0.824 | -0.000 |
| control: DenseNet + DenseNet features, logreg | `fusion/dn+dn2_feat` | 3 | 0.827 | 0.816 ± 0.004 | 0.708 | 0.776 | 0.810 | 0.821 | -0.011 |
| control: DenseNet + DenseNet + eyes features, logreg | `fusion/dn+dn2_eyes_feat` | 3 | 0.837 | 0.829 ± 0.006 | 0.722 | 0.783 | 0.822 | 0.836 | -0.013 |
| control: DenseNet + DenseNet + eyes features, MLP | `fusion/dn+dn2_eyes_mlp` | 3 | 0.840 | 0.830 ± 0.008 | 0.716 | 0.794 | 0.817 | 0.844 | -0.027 |

## ablation

| arm | run | n | val AUC | test AUC | Sens | Spec | AUC female | AUC male | F - M |
|---|---|---|---|---|---|---|---|---|---|
| final: RETFound30 + DenseNet + eyes features, MLP | `fusion/rf30+dn_eyes_mlp` | 3 | 0.844 | 0.841 ± 0.007 | 0.729 | 0.794 | 0.827 | 0.854 | -0.027 |
| without the other eye | `fusion/rf30+dn_mlp` | 3 | 0.829 | 0.822 ± 0.006 | 0.695 | 0.781 | 0.813 | 0.830 | -0.017 |
| without DenseNet | `fusion/rf30_eyes_mlp` | 3 | 0.816 | 0.813 ± 0.005 | 0.710 | 0.761 | 0.797 | 0.827 | -0.030 |
| without RETFound | `fusion/dn_eyes_mlp` | 3 | 0.824 | 0.813 ± 0.003 | 0.706 | 0.783 | 0.801 | 0.824 | -0.023 |
| other-eye block: other eye's features only | `fusion/rf30+dn_eyes-other_mlp` | 3 | 0.840 | 0.840 ± 0.006 | 0.764 | 0.764 | 0.829 | 0.849 | -0.020 |
| other-eye block: abs(own - other) only | `fusion/rf30+dn_eyes-diff_mlp` | 3 | 0.840 | 0.838 ± 0.006 | 0.730 | 0.789 | 0.824 | 0.852 | -0.027 |
| no cross-fitting (in-sample train features) | `fusion/rf30+dn_eyes_mlp_insample` | 3 | 0.835 | 0.825 ± 0.010 | 0.731 | 0.777 | 0.808 | 0.842 | -0.035 |
| no cross-fitting, logreg head | `fusion/rf30+dn_eyes_feat_insample` | 3 | 0.838 | 0.831 ± 0.005 | 0.684 | 0.821 | 0.822 | 0.840 | -0.018 |
| no cross-fitting, DenseNet + eyes, logreg head | `fusion/dn_eyes_feat_insample` | 3 | 0.824 | 0.821 ± 0.011 | 0.693 | 0.791 | 0.817 | 0.825 | -0.008 |
| without the other eye, logreg head | `fusion/rf30+dn_feat` | 3 | 0.827 | 0.824 ± 0.005 | 0.731 | 0.754 | 0.815 | 0.831 | -0.016 |
| without DenseNet, logreg head | `fusion/rf30_eyes_feat` | 3 | 0.811 | 0.815 ± 0.003 | 0.736 | 0.743 | 0.801 | 0.827 | -0.027 |
| without RETFound, logreg head | `fusion/dn_eyes_feat` | 3 | 0.823 | 0.818 ± 0.006 | 0.673 | 0.813 | 0.814 | 0.823 | -0.009 |

## adaptation

| arm | run | n | val AUC | test AUC | Sens | Spec | AUC female | AUC male | F - M |
|---|---|---|---|---|---|---|---|---|---|
| frozen + linear (RETFound recipe) | `retfound_mae_lp_none` | 3 | 0.770 | 0.768 ± 0.001 | 0.662 | 0.740 | 0.763 | 0.771 | -0.009 |
| frozen + linear (unified recipe, aug.) | `retfound_mae_lp_u_full` | 3 | 0.718 | 0.720 ± 0.007 | 0.648 | 0.672 | 0.724 | 0.715 | +0.009 |
| frozen + MLP 256 | `retfound_mae_mlp_h256` | 3 | 0.793 | 0.780 ± 0.004 | 0.683 | 0.723 | 0.774 | 0.786 | -0.013 |
| frozen + MLP 512 | `retfound_mae_mlp_h512` | 3 | 0.790 | 0.777 ± 0.006 | 0.712 | 0.716 | 0.770 | 0.784 | -0.014 |
| LoRA r8 on qkv, lr 1e-3 | `retfound_mae_lora_lr1e-3` | 3 | 0.796 | 0.797 ± 0.008 | 0.640 | 0.790 | 0.794 | 0.799 | -0.005 |
| LoRA r8 on qkv, lr 3e-4 | `retfound_mae_lora_lr3e-4` | 3 | 0.793 | 0.800 ± 0.002 | 0.631 | 0.807 | 0.801 | 0.799 | +0.002 |
| last 4 blocks | `retfound_mae_last4` | 3 | 0.794 | 0.783 ± 0.003 | 0.671 | 0.728 | 0.770 | 0.794 | -0.024 |
| LP-FT | `retfound_mae_lpft` | 3 | 0.789 | 0.788 ± 0.010 | 0.761 | 0.669 | 0.773 | 0.801 | -0.029 |
| full fine-tuning, 30 epochs | `retfound_mae_ft30_full` | 3 | 0.805 | 0.798 ± 0.003 | 0.720 | 0.715 | 0.791 | 0.805 | -0.015 |
| full fine-tuning, 100 epochs | `retfound_mae_ft_flip` | 3 | 0.805 | 0.798 ± 0.003 | 0.720 | 0.715 | 0.791 | 0.805 | -0.015 |
| full fine-tuning, unified recipe | `retfound_mae_ft_u_full` | 3 | 0.783 | 0.779 ± 0.005 | 0.629 | 0.761 | 0.772 | 0.785 | -0.013 |

## recipes

| arm | run | n | val AUC | test AUC | Sens | Spec | AUC female | AUC male | F - M |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 paper x1 none | `resnet50_none` | 3 | 0.720 | 0.704 ± 0.057 | 0.704 | 0.591 | 0.698 | 0.709 | -0.011 |
| ResNet-50 paper x1 flip | `resnet50_flip` | 3 | 0.719 | 0.703 ± 0.031 | 0.684 | 0.614 | 0.705 | 0.703 | +0.002 |
| ResNet-50 paper x1 flip_rot | `resnet50_flip_rot` | 3 | 0.717 | 0.712 ± 0.066 | 0.654 | 0.658 | 0.723 | 0.702 | +0.021 |
| ResNet-50 paper x0.1 none | `resnet50_none_lr0.1x` | 3 | 0.788 | 0.776 ± 0.001 | 0.662 | 0.742 | 0.768 | 0.782 | -0.014 |
| DenseNet-121 paper x1 none | `densenet121_none` | 3 | 0.720 | 0.711 ± 0.060 | 0.734 | 0.563 | 0.713 | 0.709 | +0.003 |
| DenseNet-121 paper x1 flip | `densenet121_flip` | 3 | 0.731 | 0.705 ± 0.027 | 0.673 | 0.625 | 0.711 | 0.700 | +0.011 |
| DenseNet-121 paper x1 flip_rot | `densenet121_flip_rot` | 3 | 0.690 | 0.680 ± 0.014 | 0.722 | 0.542 | 0.686 | 0.674 | +0.012 |
| DenseNet-121 paper x0.1 none | `densenet121_none_lr0.1x` | 3 | 0.775 | 0.767 ± 0.005 | 0.695 | 0.696 | 0.756 | 0.776 | -0.020 |
| ViT-B/16 paper x1 none | `vit_b16_none` | 3 | 0.760 | 0.767 ± 0.003 | 0.712 | 0.660 | 0.764 | 0.769 | -0.004 |
| ViT-B/16 paper x1 flip | `vit_b16_flip` | 3 | 0.765 | 0.770 ± 0.007 | 0.671 | 0.715 | 0.764 | 0.775 | -0.012 |
| ViT-B/16 paper x1 flip_rot | `vit_b16_flip_rot` | 3 | 0.775 | 0.785 ± 0.009 | 0.684 | 0.721 | 0.787 | 0.782 | +0.006 |
| ViT-B/16 paper x0.1 flip_rot | `vit_b16_flip_rot_lr0.1x` | 3 | 0.756 | 0.771 ± 0.003 | 0.710 | 0.682 | 0.778 | 0.765 | +0.013 |
| RETFound own x1 none | `retfound_mae_ft_none` | 3 | 0.790 | 0.785 ± 0.003 | 0.712 | 0.703 | 0.778 | 0.791 | -0.013 |
| RETFound own x1 flip | `retfound_mae_ft_flip` | 3 | 0.805 | 0.798 ± 0.003 | 0.720 | 0.715 | 0.791 | 0.805 | -0.015 |
| RETFound own x1 flip_rot | `retfound_mae_ft_flip_rot` | 3 | 0.803 | 0.792 ± 0.006 | 0.729 | 0.702 | 0.787 | 0.797 | -0.010 |
| RETFound frozen + linear x0.1 | `retfound_mae_lp_none_lr0.1x` | 3 | 0.750 | 0.758 ± 0.000 | 0.724 | 0.656 | 0.756 | 0.760 | -0.004 |

## paired bootstrap (test AUC, patient-level, 2000 resamples)

| A | B | A - B | 95% CI | p |
|---|---|---|---|---|
| `retfound_mae_ft_flip` | `densenet121_ft_full` | +0.0052 | [-0.0147, +0.0246] | 0.6090 |
| `retfound_mae_ft_flip` | `vit_b16_ft_full` | +0.0316 | [+0.0134, +0.0492] | 0.0005 |
| `fusion/rf+dn` | `fusion/dn_seedens` | +0.0013 | [-0.0105, +0.0134] | 0.8730 |
| `fusion/rf+eyes` | `retfound_mae_ft_flip` | +0.0135 | [+0.0074, +0.0201] | 0.0005 |
| `fusion/dn+eyes` | `densenet121_ft_full` | +0.0169 | [+0.0097, +0.0244] | 0.0005 |
| `fusion/rf+dn+eyes` | `densenet121_ft_full` | +0.0424 | [+0.0295, +0.0555] | 0.0005 |
| `fusion/rf+dn+eyes` | `fusion/dn_seedens+eyes` | +0.0037 | [-0.0075, +0.0153] | 0.5860 |
| `fusion/rf30+dn_feat` | `fusion/rf30+dn` | +0.0046 | [-0.0005, +0.0098] | 0.0710 |
| `fusion/rf30+dn_eyes_feat` | `fusion/rf30+dn+eyes` | +0.0028 | [-0.0042, +0.0100] | 0.4470 |
| `fusion/rf30+dn_eyes_feat` | `densenet121_ft_full` | +0.0452 | [+0.0299, +0.0605] | 0.0005 |
| `fusion/dn_eyes_feat` | `fusion/dn+eyes` | +0.0080 | [+0.0004, +0.0152] | 0.0360 |
| `retfound_mae_lora_lr1e-3` | `retfound_mae_ft30_full` | -0.0016 | [-0.0130, +0.0104] | 0.7950 |
| `retfound_mae_last4` | `retfound_mae_ft30_full` | -0.0156 | [-0.0248, -0.0057] | 0.0020 |
| `retfound_mae_lpft` | `retfound_mae_ft30_full` | -0.0107 | [-0.0215, -0.0000] | 0.0500 |
| `retfound_mae_lora_lr3e-4` | `retfound_mae_ft30_full` | +0.0016 | [-0.0124, +0.0158] | 0.7830 |
| `retfound_mae_mlp_h256` | `retfound_mae_lp_none` | +0.0128 | [-0.0021, +0.0281] | 0.0930 |
| `fusion/rf30+dn` | `fusion/dn+dn2` | +0.0091 | [-0.0019, +0.0206] | 0.1170 |
| `fusion/rf30+dn+eyes` | `fusion/dn+dn2+eyes` | +0.0118 | [+0.0012, +0.0228] | 0.0310 |
| `fusion/rf30+dn_feat` | `fusion/dn+dn2_feat` | +0.0081 | [-0.0041, +0.0204] | 0.2110 |
| `fusion/rf30+dn_eyes_feat` | `fusion/dn+dn2_eyes_feat` | +0.0093 | [-0.0021, +0.0210] | 0.1180 |
| `fusion/rf30+dn_eyes_mlp` | `fusion/dn+dn2_eyes_mlp` | +0.0103 | [-0.0006, +0.0214] | 0.0640 |
| `fusion/rf30+dn_eyes_mlp` | `fusion/rf30+dn_eyes_feat` | +0.0024 | [-0.0011, +0.0062] | 0.1930 |
| `fusion/rf30+dn_eyes_mlp` | `densenet121_ft_full` | +0.0476 | [+0.0329, +0.0624] | 0.0005 |
| `fusion/rf30+dn_eyes_mlp` | `retfound_mae_ft_flip` | +0.0424 | [+0.0301, +0.0548] | 0.0005 |
| `fusion/rf30+dn_eyes_mlp` | `fusion/rf30+dn_mlp` | +0.0190 | [+0.0110, +0.0274] | 0.0005 |
| `fusion/rf30+dn_eyes_mlp` | `fusion/rf30_eyes_mlp` | +0.0277 | [+0.0178, +0.0383] | 0.0005 |
| `fusion/rf30+dn_eyes_mlp` | `fusion/dn_eyes_mlp` | +0.0276 | [+0.0165, +0.0390] | 0.0005 |
| `fusion/rf30_eyes_mlp` | `fusion/dn_eyes_mlp` | -0.0001 | [-0.0186, +0.0175] | 0.9920 |
| `fusion/rf30+dn_eyes_mlp` | `fusion/rf30+dn_eyes-other_mlp` | +0.0012 | [-0.0024, +0.0048] | 0.5120 |
| `fusion/rf30+dn_eyes_mlp` | `fusion/rf30+dn_eyes-diff_mlp` | +0.0023 | [-0.0020, +0.0061] | 0.2950 |
| `fusion/rf30+dn_eyes_mlp` | `fusion/rf30+dn_eyes_mlp_insample` | +0.0152 | [+0.0061, +0.0249] | 0.0010 |
| `fusion/rf30+dn_eyes_feat` | `fusion/rf30+dn_eyes_feat_insample` | +0.0069 | [-0.0011, +0.0152] | 0.1040 |
| `fusion/dn_eyes_feat` | `fusion/dn_eyes_feat_insample` | -0.0026 | [-0.0090, +0.0041] | 0.4190 |
| `fusion/rf30+dn_eyes_mlp` | `fusion/rf30+dn+eyes` | +0.0052 | [-0.0017, +0.0123] | 0.1350 |
| `fusion/rf30+dn_eyes_mlp_insample` | `fusion/rf30+dn+eyes` | -0.0100 | [-0.0206, -0.0005] | 0.0420 |
| `fusion/rf30+dn_mlp` | `fusion/rf30+dn_feat` | -0.0018 | [-0.0043, +0.0005] | 0.1230 |
| `fusion/rf30_eyes_mlp` | `fusion/rf30_eyes_feat` | -0.0019 | [-0.0055, +0.0018] | 0.3070 |
| `fusion/dn_eyes_mlp` | `fusion/dn_eyes_feat` | -0.0049 | [-0.0109, +0.0011] | 0.1210 |
