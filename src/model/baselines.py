"""Baseline classifiers (timm ImageNet weights) with the optimizer and LR schedule of each model's original paper.

Hyperparameters live in config/baselines/<model>.yaml; this module only turns them into objects.
"""
import math

import timm
import torch
from timm.data import resolve_data_config

from src.model.pfm import apply_lora, build_retfound_classifier, set_trainable


def build_baseline(model_cfg, cache_dir=None, pretrained=True):
    """Pretrained timm model with a fresh 1-logit head, plus the timm data config (mean, std, interpolation)
    its weights expect. Weights are downloaded from the Hugging Face Hub into cache_dir on first use."""
    kwargs = {k: model_cfg[k] for k in ("drop_rate", "drop_path_rate") if k in model_cfg}
    model = timm.create_model(model_cfg["timm_name"], pretrained=pretrained, num_classes=1,
                              cache_dir=cache_dir, **kwargs)
    if model_cfg.get("zero_init_head"):
        head = model.get_classifier()
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    return model, resolve_data_config({}, model=model)


def build_model(model_cfg, cache_dir, pretrained=True):
    """The classifier a model config describes: configs with hf_repo (RETFound) load their own checkpoint, the others
    timm ImageNet weights; then optional LoRA (model.lora) and the trainable parameters (model.trainable: "all", "head",
    "lora" or the number of last ViT blocks; default "all", "head" for freeze_backbone, "lora" with LoRA)."""
    build = build_retfound_classifier if "hf_repo" in model_cfg else build_baseline
    model, data_cfg = build(model_cfg, cache_dir=cache_dir, pretrained=pretrained)
    lora = model_cfg.get("lora")
    if lora:
        apply_lora(model, lora["rank"], lora["alpha"], lora["dropout"])
    set_trainable(model, model_cfg.get("trainable", "head" if model_cfg.get("freeze_backbone") else "lora" if lora else "all"))
    return model, data_cfg


def build_optimizer(model, optim_cfg, batch_size):
    """Paper optimizer for the trainable parameters of `model`, with the paper lr scaled linearly from its batch size
    to ours (Goyal et al., 2017)."""
    lr = optim_cfg["base_lr"] * batch_size / optim_cfg["base_batch_size"]
    if optim_cfg["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=optim_cfg["momentum"],
                               nesterov=optim_cfg.get("nesterov", False), weight_decay=optim_cfg["weight_decay"])
    if optim_cfg["name"] == "adamw":
        # MAE / RETFound style: no weight decay on 1-D tensors (biases, norms) or the model's no_weight_decay() names;
        # with layer_decay, each ViT layer gets lr x layer_decay^(depth from the top) (BEiT / MAE param_groups_lrd).
        groups = vit_param_groups(model, optim_cfg["weight_decay"], optim_cfg.get("layer_decay"))
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(optim_cfg.get("betas", (0.9, 0.999))))
    raise ValueError(f"unknown optimizer {optim_cfg['name']!r}")


def vit_param_groups(model, weight_decay, layer_decay=None):
    """Port of util/lr_decay.py:param_groups_lrd from MAE / RETFound. Without layer_decay every group has lr_scale 1.

    Layer ids: cls_token, pos_embed, patch_embed -> 0; blocks.i -> i + 1; everything else (fc_norm, head) -> depth + 1.
    """
    no_wd = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    num_layers = len(model.blocks) + 1 if layer_decay is not None else 0
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = not (p.ndim == 1 or n in no_wd)
        if layer_decay is None:
            layer_id, scale = 0, 1.0
        else:
            if n in ("cls_token", "pos_embed") or n.startswith("patch_embed"):
                layer_id = 0
            elif n.startswith("blocks."):
                layer_id = int(n.split(".")[1]) + 1
            else:
                layer_id = num_layers
            scale = layer_decay ** (num_layers - layer_id)
        g = groups.setdefault((layer_id, decay), {"params": [], "weight_decay": weight_decay if decay else 0.0,
                                                  "lr_scale": scale})
        g["params"].append(p)
    return list(groups.values())


class LRSchedule:
    """The paper schedules behind one interface.

    plateau:       lr x factor when val AUC has not improved for `patience` epochs (ResNet)
    multistep:     lr x factor at fixed fractions of training (DenseNet)
    cosine:        cosine decay to 0 over all steps, optional linear warmup (ViT fine-tuning)
    warmup_cosine: linear warmup from 0 for `warmup_epochs`, then half-cycle cosine to `min_lr`, on the fractional
                   epoch (MAE / RETFound util/lr_sched.py)

    Call before_step() before every optimizer step and end_epoch() after every validation.
    """

    def __init__(self, optimizer, sched_cfg, epochs, steps_per_epoch):
        self.opt = optimizer
        self.name = sched_cfg["name"]
        self.cfg = sched_cfg
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.epochs, self.steps_per_epoch = epochs, steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.step_num = 0
        if self.name == "plateau":
            self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=sched_cfg["factor"], patience=sched_cfg["patience"])
        elif self.name == "multistep":
            self.milestones = [round(f * epochs) for f in sched_cfg["milestones"]]
        elif self.name not in ("cosine", "warmup_cosine"):
            raise ValueError(f"unknown schedule {self.name!r}")

    def _set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)

    def before_step(self):
        if self.name == "cosine":
            s, warm = self.step_num, self.cfg.get("warmup_steps", 0)
            if s < warm:
                lr = self.base_lr * (s + 1) / warm
            else:
                lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * (s - warm) / max(1, self.total_steps - warm)))
            self._set_lr(lr)
        elif self.name == "warmup_cosine":
            e, warm, min_lr = self.step_num / self.steps_per_epoch, self.cfg["warmup_epochs"], self.cfg["min_lr"]
            if e < warm:
                lr = self.base_lr * e / warm
            else:
                lr = min_lr + (self.base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (e - warm) / (self.epochs - warm)))
            self._set_lr(lr)
        self.step_num += 1

    def end_epoch(self, epoch, val_metric):
        """epoch: number of epochs completed so far (1-based)."""
        if self.name == "plateau":
            self.plateau.step(val_metric)
        elif self.name == "multistep":
            self._set_lr(self.base_lr * self.cfg["factor"] ** sum(epoch >= m for m in self.milestones))

    def state_dict(self):
        state = {"step_num": self.step_num, "base_lr": self.base_lr}
        if self.name == "plateau":
            state["plateau"] = self.plateau.state_dict()
        return state

    def load_state_dict(self, state):
        self.step_num, self.base_lr = state["step_num"], state["base_lr"]
        if self.name == "plateau":
            self.plateau.load_state_dict(state["plateau"])

    def summary(self):
        """Readable schedule position, for best.json."""
        out = {"name": self.name, "base_lr": self.base_lr, "steps_done": self.step_num}
        if self.name == "plateau":
            out.update(best_val_metric=self.plateau.best, epochs_without_improvement=self.plateau.num_bad_epochs)
        elif self.name == "multistep":
            out["milestone_epochs"] = self.milestones
        else:
            out["total_steps"] = self.total_steps
            if self.name == "warmup_cosine":
                out.update(warmup_epochs=self.cfg["warmup_epochs"], min_lr=self.cfg["min_lr"])
        return out
