"""ODIR-5K diabetic retinopathy data: npz loading and dataloaders.

Each npz file holds one eye: `slo_fundus` (200x200x3 uint8 color fundus photo, stored BGR), `dr_class` (0/1)
and `male` (0/1). Files are named <patient>_<left|right>.npz. The official train/val/test folders are used as given.
"""
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T

SPLITS = ("train", "val", "test")
AUGMENTS = ("none", "flip", "flip_rot", "full")
# flip_rot: flip, then rotate by up to data.rotate_deg.
# full: random resized crop (data.crop_scale, data.crop_ratio), flip, rotation, brightness/contrast jitter (data.jitter).


class ODIRDataset(Dataset):
    """All images of one split, loaded into memory once (about 120 KB per image)."""

    def __init__(self, split_dir, transform=None, limit=None, seed=0, keep=None):
        """keep: optional set of image ids (e.g. "2707_left") to use instead of the whole folder."""
        files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        if keep is not None:
            files = [f for f in files if os.path.basename(f)[:-4] in keep]
        if not files:
            raise FileNotFoundError(f"no npz files in {split_dir}")
        if limit is not None and limit < len(files):
            rng = np.random.default_rng(seed)
            files = sorted(rng.choice(files, size=limit, replace=False))

        images, labels, male = [], [], []
        for f in files:
            with np.load(f) as d:
                images.append(d["slo_fundus"][..., ::-1])  # BGR -> RGB
                labels.append(int(d["dr_class"]))
                male.append(int(d["male"]))
        self.images = np.ascontiguousarray(np.stack(images))  # (N, H, W, 3) uint8 RGB
        self.labels = np.asarray(labels, dtype=np.int64)
        self.male = np.asarray(male, dtype=np.int64)
        self.ids = [os.path.basename(f)[:-4] for f in files]  # e.g. "2707_left"
        self.patients = [i.rsplit("_", 1)[0] for i in self.ids]
        self.eyes = [i.rsplit("_", 1)[1] for i in self.ids]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1)  # (3, H, W) uint8
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "label": torch.tensor(self.labels[i], dtype=torch.float32),
            "male": int(self.male[i]),
            "id": self.ids[i],
        }


def build_transform(data_cfg, image_size):
    """Resize to the model's input size (no crop, so the whole fundus is kept), scale to [0, 1]
    and normalize with the mean/std the pretrained weights were trained with.
    Augmentation is applied per batch afterwards, see random_hflip.

    data_cfg: timm data config of the model, with `mean`, `std` and `interpolation`.
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode(data_cfg["interpolation"]), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
    ])


def random_hflip(images, generator):
    """Flip each image of a (B, C, H, W) batch left-right with probability 0.5.

    Runs on the batch with a generator the caller seeds per epoch, so the flips depend only on (seed, epoch),
    not on the number of loader workers or on whether the run was resumed.
    """
    flip = torch.rand(images.shape[0], generator=generator, device=images.device) < 0.5
    return torch.where(flip[:, None, None, None], images.flip(-1), images)


def random_rotate(images, generator, max_deg, fill):
    """Rotate each image of a (B, C, H, W) batch about its center by an angle drawn uniformly from [-max_deg, max_deg],
    with the same per-epoch seeding as random_hflip. Pixels rotated in from outside the image get `fill`, a per-channel
    value (the normalized value of black, so the corners stay background)."""
    angle = (torch.rand(images.shape[0], generator=generator, device=images.device) * 2 - 1) * math.radians(max_deg)
    cos, sin, zero = torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)
    theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    fill = fill.view(1, -1, 1, 1)
    return F.grid_sample(images - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + fill


def random_resized_crop(images, generator, scale, ratio):
    """Crop a random box from each image of a (B, C, H, W) batch and resize it back to H x W, like torchvision
    RandomResizedCrop: area fraction ~ U(scale), aspect ratio log-uniform in `ratio`, position uniform inside the image.
    The box is clipped to the image, so no padding is needed."""
    b, dev = images.shape[0], images.device
    u = torch.rand(b, 4, generator=generator, device=dev)
    area = scale[0] + (scale[1] - scale[0]) * u[:, 0]
    log_r = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * u[:, 1]
    w = torch.sqrt(area * torch.exp(log_r)).clamp(max=1.0)  # box width / image width
    h = torch.sqrt(area / torch.exp(log_r)).clamp(max=1.0)
    cx, cy = (1 - w) * (2 * u[:, 2] - 1), (1 - h) * (2 * u[:, 3] - 1)  # box center in [-1, 1] coordinates
    zero = torch.zeros_like(w)
    theta = torch.stack([torch.stack([w, zero, cx], 1), torch.stack([zero, h, cy], 1)], 1)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def color_jitter(images, generator, strength, mean, std):
    """Brightness then contrast jitter on a normalized (B, C, H, W) batch, factors ~ U(1 - strength, 1 + strength) per
    image, as torchvision ColorJitter(brightness, contrast) in a fixed order. Works in [0, 1] pixel space: brightness
    scales the image, contrast blends it with the mean of its grayscale version; mean/std: (1, C, 1, 1) tensors."""
    b = images.shape[0]
    f = 1 + strength * (2 * torch.rand(b, 2, generator=generator, device=images.device) - 1)
    x = (images * std + mean) * f[:, 0].view(b, 1, 1, 1)
    x = x.clamp(0, 1)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean(dim=(1, 2)).view(b, 1, 1, 1)
    x = (f[:, 1].view(b, 1, 1, 1) * x + (1 - f[:, 1].view(b, 1, 1, 1)) * gray).clamp(0, 1)
    return (x - mean) / std


def build_dataloaders(root, data_cfg, image_size, batch_size, num_workers=4, limit=None, seed=0, holdout=None):
    """Loaders for the official train/val/test folders. Only train is shuffled, by a sampler whose generator
    (loaders["train"].sampler.generator) the caller re-seeds every epoch, so the order of an epoch does not
    depend on the epochs before it.

    limit: use a random subset of this many images per split (for smoke tests).
    holdout: set of train image ids (one cross-fitting fold) left out of training and served as an extra "oof" loader.
    """
    transform = build_transform(data_cfg, image_size)
    train_ids = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(root, "train", "*.npz"))}
    keep = {"train": train_ids - holdout, "oof": holdout} if holdout is not None else {}
    loaders = {}
    for split in SPLITS + (("oof",) if holdout is not None else ()):
        folder = "train" if split == "oof" else split
        ds = ODIRDataset(os.path.join(root, folder), transform, limit=limit, seed=seed, keep=keep.get(split))
        sampler = RandomSampler(ds, generator=torch.Generator()) if split == "train" else None
        loaders[split] = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                                    pin_memory=True, persistent_workers=num_workers > 0)
    return loaders
