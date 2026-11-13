"""标准 VGG16、强增强与正确的混合标签；本模块不读取测试答案。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass
class Config:
    image_size: int = 128  # 用户最新确认：恢复 128x128，先实测单卡与双卡速度。
    batch_size: int = 96
    eval_batch_size: int = 192
    max_epochs: int = 20
    min_epochs: int = 10
    epochs: int = 20
    warmup_epochs: int = 2
    patience: int = 5
    lr: float = 2.5e-4
    head_lr_multiplier: float = 3.0
    weight_decay: float = 5e-4
    dropout: float = 0.35
    label_smoothing: float = 0.05
    ema_decay: float = 0.995
    ema_warmup_epochs: int = 1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_probability: float = 0.3
    cutmix_probability: float = 0.3
    unmixed_final_epochs: int = 2
    crop_scale: tuple = (0.72, 1.0)
    crop_ratio: tuple = (0.9, 1.1)
    rotation: float = 15.0
    jitter: float = 0.25
    erasing_probability: float = 0.1
    attention: str = "none"  # 可设 eca；本轮优先检验五折和泛化训练策略。
    workers: int = 2
    cpu_threads: int = 2
    seed: int = 42
    total_budget_seconds: int = 10200  # 2 小时 50 分钟，包含预处理和预测。
    finish_reserve_seconds: int = 600
    throughput_safety_factor: float = 1.3
    input_mean: tuple = MEAN
    input_std: tuple = STD
    interpolation: str = "bilinear"

    def validate(self):
        assert self.image_size >= 32 and self.batch_size >= 2
        assert self.eval_batch_size >= 1 and 1 <= self.min_epochs <= self.max_epochs
        assert 1 <= self.epochs <= self.max_epochs and self.warmup_epochs >= 0
        assert self.lr > 0 and self.head_lr_multiplier > 0 and self.weight_decay >= 0
        assert 0 <= self.dropout < 1 and 0 <= self.label_smoothing < 1
        assert 0 <= self.ema_decay < 1 and self.patience >= 0
        assert self.mixup_alpha > 0 and self.cutmix_alpha > 0
        assert self.mixup_probability >= 0 and self.cutmix_probability >= 0
        assert self.mixup_probability + self.cutmix_probability <= 1
        assert self.unmixed_final_epochs >= 0 and self.workers >= 0
        assert self.attention in {"none", "eca"}
        assert 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1
        assert 0 < self.crop_ratio[0] <= self.crop_ratio[1]
        assert len(self.input_mean) == len(self.input_std) == 3
        assert all(value > 0 for value in self.input_std)
        assert self.interpolation in {"bilinear", "bicubic"}
        return self

    def dictionary(self):
        return asdict(self)


def seed_everything(seed, cpu_threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 固定尺寸 + AMP 优先保证吞吐；设置种子不等于逐位确定性。
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def make_transform(config, training):
    operations = []
    interpolation = (InterpolationMode.BICUBIC if config.interpolation == "bicubic"
                     else InterpolationMode.BILINEAR)
    if training:
        # 面部图已裁剪：限制最小面积和宽高比，减少切掉整个眉眼或嘴部的风险。
        operations.extend([
            T.RandomResizedCrop(config.image_size, scale=config.crop_scale,
                                ratio=config.crop_ratio, interpolation=interpolation),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(config.rotation, interpolation=InterpolationMode.BILINEAR, fill=128),
            # 原图为灰度，饱和度/hue 不会带来真实色彩变化，故只调整亮度和对比度。
            T.ColorJitter(brightness=config.jitter, contrast=config.jitter),
        ])
    else:
        operations.append(T.Resize((config.image_size, config.image_size),
                                   interpolation=interpolation, antialias=True))
    # 不同预训练模型的输入分布不同：MobileViT 使用 0 均值/1 标准差，即保持 [0,1]。
    operations.extend([T.ToTensor(), T.Normalize(config.input_mean, config.input_std)])
    if training and config.erasing_probability:
        operations.append(T.RandomErasing(config.erasing_probability, scale=(0.02, 0.08),
                                          ratio=(0.5, 2.0), value=0))
    return T.Compose(operations)


class EmotionDataset(Dataset):
    def __init__(self, array_path, indices, labels, transform):
        self.array_path = str(array_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self._images = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def __getitem__(self, position):
        if self._images is None:
            self._images = np.load(self.array_path, mmap_mode="r")
        index = int(self.indices[position])
        image = Image.fromarray(self._images[index]).convert("RGB")
        target = -1 if self.labels is None else int(self.labels[index])
        return self.transform(image), target


def make_loader(dataset, config, training, device, seed):
    # 显式 spawn 避免 Linux fork 继承 CUDA/线程状态；超时使数据阻塞能被定位。
    process_options = ({"multiprocessing_context": "spawn", "timeout": 60}
                       if config.workers > 0 else {})
    return DataLoader(dataset, batch_size=config.batch_size if training else config.eval_batch_size,
                      shuffle=training, drop_last=False, num_workers=config.workers,
                      pin_memory=device.type == "cuda", persistent_workers=config.workers > 0,
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
                      **process_options)


class ECA(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        size = int(abs((math.log2(channels) + 1) / 2))
        size = size if size % 2 else size + 1
        self.conv = nn.Conv1d(1, 1, size, padding=size // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, images):
        gate = 2 * self.conv(images.mean((2, 3)).unsqueeze(1)).sigmoid()
        return images * gate.squeeze(1)[:, :, None, None]


class VGG16(nn.Module):
    def __init__(self, config, weights=None):
        super().__init__()
        # 标准 VGG16 configuration D：13 个卷积层、5 个池化层；省去不使用的 4096 维分类头。
        configuration = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                         512, 512, 512, "M", 512, 512, 512, "M"]
        layers, incoming = [], 3
        for outgoing in configuration:
            if outgoing == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layer = nn.Conv2d(incoming, outgoing, 3, padding=1)
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
                layers.extend([layer, nn.ReLU(inplace=True)])
                incoming = outgoing
        self.features = nn.Sequential(*layers)
        if weights is not None:
            state = torch.load(Path(weights), map_location="cpu", weights_only=True)
            convolution = {key.removeprefix("features."): value for key, value in state.items()
                           if key.startswith("features.")}
            self.features.load_state_dict(convolution, strict=True)
            del state, convolution
        self.attention = ECA() if config.attention == "eca" else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(512, 6))

    def forward(self, images):
        # ECA 若启用，放在最后一个池化前；默认路径仍为标准 VGG16 + GAP。
        features = self.features[:-1](images)
        features = self.features[-1](self.attention(features))
        return self.head(self.pool(features).flatten(1))


def mix_batch(images, labels, config, forced=None):
    """整批选择 MixUp、CutMix 或原图；返回对应两套标签和真实混合比例。"""
    if forced is None:
        choice = np.random.random()
        forced = ("cutmix" if choice < config.cutmix_probability else
                  "mixup" if choice < config.cutmix_probability + config.mixup_probability else "none")
    if forced == "none" or len(images) < 2:
        return images, labels, labels, 1.0, "none"
    permutation = torch.randperm(len(images), device=images.device)
    if forced == "mixup":
        ratio = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
        return ratio * images + (1 - ratio) * images[permutation], labels, labels[permutation], ratio, forced
    if forced != "cutmix":
        raise ValueError(forced)
    ratio = float(np.random.beta(config.cutmix_alpha, config.cutmix_alpha))
    height, width = images.shape[-2:]
    side = math.sqrt(1 - ratio)
    box_height, box_width = max(1, int(height * side)), max(1, int(width * side))
    center_y, center_x = np.random.randint(height), np.random.randint(width)
    y0, y1 = max(0, center_y - box_height // 2), min(height, center_y + (box_height + 1) // 2)
    x0, x1 = max(0, center_x - box_width // 2), min(width, center_x + (box_width + 1) // 2)
    mixed = images.clone()
    mixed[:, :, y0:y1, x0:x1] = images[permutation, :, y0:y1, x0:x1]
    # 裁到图像边缘后，必须用实际粘贴面积重算 lambda，而非使用 Beta 抽样值。
    ratio = 1.0 - ((y1 - y0) * (x1 - x0)) / (height * width)
    return mixed, labels, labels[permutation], ratio, forced


def mixed_loss(logits, first_labels, second_labels, ratio, config):
    return (ratio * F.cross_entropy(logits, first_labels, label_smoothing=config.label_smoothing)
            + (1 - ratio) * F.cross_entropy(logits, second_labels, label_smoothing=config.label_smoothing))
