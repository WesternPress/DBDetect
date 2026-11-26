"""Pathology/ophthalmic foundation model encoders: frozen feature extractor (linear probe) and trainable classifier
(fine-tuning).

RETFound (MAE) is a timm ViT-L/16 with global average pooling over the patch tokens followed by a LayerNorm (fc_norm),
exactly as `models_vit.RETFound_mae(global_pool=True)` in the official repo. The MAE checkpoint has no fc_norm (it was
pre-trained with the final `norm` on all tokens), so fc_norm starts from its initialization (weight 1, bias 0).
"""
import math

import timm
import torch
from huggingface_hub import hf_hub_download
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

RETFOUND_DATA_CFG = {"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD, "interpolation": "bicubic"}


def _load_retfound(model, model_cfg, cache_dir, new_keys):
    """Load the MAE encoder weights into a timm ViT and check that only `new_keys` are missing and only the
    pre-training final norm and MAE decoder parts are left over. The weights are gated on the Hugging Face Hub and
    must already be in cache_dir (downloaded once with a token)."""
    path = hf_hub_download(model_cfg["hf_repo"], model_cfg["hf_file"], cache_dir=cache_dir)
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    msg = model.load_state_dict(state, strict=False)
    if set(msg.missing_keys) != set(new_keys):
        raise RuntimeError(f"unexpected missing keys: {msg.missing_keys}")
    leftover = [k for k in msg.unexpected_keys
                if k not in ("norm.weight", "norm.bias", "mask_token") and not k.startswith("decoder")]
    if leftover:
        raise RuntimeError(f"unexpected checkpoint keys: {leftover}")
    return {"checkpoint": path, "missing_keys": msg.missing_keys, "unexpected_keys": msg.unexpected_keys}


def build_retfound(model_cfg, cache_dir):
    """Frozen RETFound encoder returning one 1024-d feature per image, the data config its weights expect, and
    information about the loaded checkpoint."""
    model = timm.create_model(model_cfg["timm_arch"], pretrained=False, num_classes=0, global_pool="avg")
    info = _load_retfound(model, model_cfg, cache_dir, ["fc_norm.weight", "fc_norm.bias"])
    model.requires_grad_(False).eval()
    return model, dict(RETFOUND_DATA_CFG), info


def build_retfound_classifier(model_cfg, cache_dir, pretrained=True):
    """RETFound with a 1-logit head for fine-tuning, initialized like the official repo: drop path from the config,
    head weight ~ trunc_normal(std=head_init_std), bias 0. pretrained=False skips loading (e.g. before a resume)."""
    model = timm.create_model(model_cfg["timm_arch"], pretrained=False, num_classes=1, global_pool="avg",
                              drop_path_rate=model_cfg.get("drop_path_rate", 0.0))
    if pretrained:
        _load_retfound(model, model_cfg, cache_dir, ["fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"])
    torch.nn.init.trunc_normal_(model.head.weight, std=model_cfg["head_init_std"])
    torch.nn.init.zeros_(model.head.bias)
    return model, dict(RETFOUND_DATA_CFG)


class LoRALinear(torch.nn.Module):
    """Low-rank adaptation of a frozen linear layer (Hu et al., 2021): y = W x + (alpha / rank) * B A dropout(x).
    B starts at zero, so the wrapped layer computes exactly the original output at initialization."""

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora_a = torch.nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = torch.nn.Linear(rank, base.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)
        self.drop = torch.nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(self.drop(x))) * self.scale


def apply_lora(model, rank, alpha, dropout):
    """Wrap the attention qkv projection of every ViT block with LoRALinear."""
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, rank, alpha, dropout)


def set_trainable(model, spec):
    """Choose which parameters train. spec: "all"; "head" (linear probe); "lora" (LoRA matrices + fc_norm + head);
    or an int k (the last k ViT blocks + fc_norm + head; everything before them frozen)."""
    if spec == "all":
        rule = lambda n: True
    elif spec == "head":
        rule = lambda n: n.startswith("head")
    elif spec == "lora":
        rule = lambda n: "lora_" in n or n.startswith(("head", "fc_norm"))
    elif isinstance(spec, int):
        first = len(model.blocks) - spec
        rule = lambda n: n.startswith(("head", "fc_norm")) or (n.startswith("blocks.") and int(n.split(".")[1]) >= first)
    else:
        raise ValueError(f"unknown trainable spec {spec!r}")
    for n, p in model.named_parameters():
        p.requires_grad = rule(n)
