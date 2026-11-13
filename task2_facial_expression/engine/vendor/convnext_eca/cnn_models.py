"""CNN 与卷积/注意力混合模型对照；预训练权重和输入归一化逐项核验。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from model_and_aug import Config as BaseConfig, ECA, VGG16


@dataclass
class Config(BaseConfig):
    architecture: str = "vgg16"
    head: str = "gap"
    loss_kind: str = "ce"
    gce_q: float = 0.7
    gce_weight: float = 0.5
    robust_start_epoch: int = 5
    selection: str = "ema"
    clean_train_samples: int = 600
    head_lr: float | None = None

    def validate(self):
        super().validate()
        assert self.architecture in {"vgg16", "resnet18", "efficientnet_b0", "mobilevit_xs", "mobilevitv2_100", "convnext_tiny"}
        assert self.head in {"gap", "spatial"}
        assert self.loss_kind in {"ce", "ce_gce", "focal"}
        assert 0 < self.gce_q <= 1 and 0 <= self.gce_weight <= 1
        assert self.selection in {"ema", "ordinary"}
        assert self.clean_train_samples >= 6
        assert self.selection != "ema" or self.ema_decay > 0
        assert self.head_lr is None or self.head_lr > 0
        return self


WEIGHTS = {
    "vgg16": ("https://download.pytorch.org/models/vgg16-397923af.pth", "397923af"),
    "resnet18": ("https://download.pytorch.org/models/resnet18-f37072fd.pth", "f37072fd"),
    "efficientnet_b0": ("https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth", "7f5810bc"),
    "convnext_tiny": ("https://download.pytorch.org/models/convnext_tiny-983f1562.pth", "983f1562"),
}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def ensure_weights(architecture, directory, existing=None):
    if architecture.startswith("mobilevit"):
        from fetch_mobilevit_weights import fetch
        return fetch(architecture, directory)[0]
    url, prefix = WEIGHTS[architecture]
    target = Path(directory) / url.rsplit("/", 1)[-1]
    candidates = [target] + ([Path(existing)] if existing else [])
    for candidate in candidates:
        if candidate.is_file() and file_hash(candidate).startswith(prefix):
            return candidate
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    # 限制连接和连续读取等待，避免无日志的权重下载阻塞。
    print(f"PHASE weights_start architecture={architecture}", flush=True)
    started, last_print, received = time.monotonic(), time.monotonic(), 0
    with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as handle:
        while block := response.read(4 * 1024 * 1024):
            handle.write(block)
            received += len(block)
            if time.monotonic() - last_print > 10:
                print(f"PHASE weights_downloading bytes={received}", flush=True)
                last_print = time.monotonic()
            if time.monotonic() - started > 180:
                raise TimeoutError("权重下载超过三分钟")
    assert file_hash(partial).startswith(prefix), "官方权重 SHA256 校验失败"
    partial.replace(target)
    print(f"PHASE weights_ready architecture={architecture}", flush=True)
    return target


class CNN(nn.Module):
    def __init__(self, config, weights=None):
        super().__init__()
        self.config = config
        self.head_norm = nn.Identity()
        if config.architecture == "vgg16":
            simple = deepcopy(config)
            simple.attention = "none"
            reference = VGG16(simple, weights)
            self.features = reference.features
            channels = 512
        elif config.architecture.startswith("mobilevit"):
            import timm
            from safetensors.torch import load_file

            backbone = timm.create_model(config.architecture + ".cvnets_in1k", pretrained=False)
            data_config = timm.data.resolve_model_data_config(backbone)
            assert tuple(config.input_mean) == tuple(data_config["mean"]), "预处理均值与预训练权重不符"
            assert tuple(config.input_std) == tuple(data_config["std"]), "预处理标准差与预训练权重不符"
            if weights is not None:
                # 先完整加载 1000 类权重；严格加载能发现转换权重/架构版本不匹配。
                backbone.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
            channels = backbone.num_features
            self.features = nn.Sequential(backbone.stem, backbone.stages, backbone.final_conv)
        else:
            backbone = getattr(models, config.architecture)(weights=None)
            if weights is not None:
                backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
            if config.architecture == "resnet18":
                # 保留已预训练的 7x7 卷积；移除紧随其后的池化，将末层空间分辨率提高到 8x8。
                backbone.maxpool = nn.Identity()
                self.features = nn.Sequential(*list(backbone.children())[:-2])
                channels = 512
            elif config.architecture == "convnext_tiny":
                self.features = backbone.features
                # ConvNeXt 原分类头在池化后做 LayerNorm；迁移时保留这层预训练归一化。
                self.head_norm = backbone.classifier[0]
                channels = backbone.classifier[-1].in_features
            else:
                self.features = backbone.features
                channels = 1280
        self.attention = ECA() if config.attention == "eca" else nn.Identity()
        if config.head == "spatial":
            # 2x2 区域表示保留粗略眉眼/嘴部位置，参数量远小于 VGG 原始全连接层。
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(config.dropout),
                nn.Linear(channels * 4, 256), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(256, 6))
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            # VGG 参照组复用原分类头初始化，避免比较增强时额外改变初始权重。
            self.classifier = (nn.Sequential(nn.Flatten(), *list(reference.head.children()))
                               if config.architecture == "vgg16" else
                               nn.Sequential(nn.Flatten(), nn.Dropout(config.dropout), nn.Linear(channels, 6)))

    def forward(self, images):
        if self.config.architecture == "vgg16":
            # 保持与参照版本一致的注意力插入位置：末次最大池化之前。
            value = self.features[:-1](images)
            value = self.features[-1](self.attention(value))
        else:
            value = self.attention(self.features(images))
        return self.classifier(self.head_norm(self.pool(value)))


def single_loss(logits, labels, config, epoch):
    # 损失在 float32 中计算，避免较小概率在 AMP 下溢。
    logits = logits.float()
    ce = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing, reduction="none")
    if config.loss_kind == "ce" or (config.loss_kind == "ce_gce" and epoch <= config.robust_start_epoch):
        return ce.mean()
    selected = logits.softmax(1).gather(1, labels[:, None]).squeeze(1).clamp_min(1e-7)
    if config.loss_kind == "focal":
        return ((1 - selected).pow(1.5) * ce).mean()
    # GCE 降低极难样本的梯度贡献；真实标签是否有噪声仍需验证，不能直接删除难样本。
    robust = (1 - selected.pow(config.gce_q)) / config.gce_q
    return ((1 - config.gce_weight) * ce + config.gce_weight * robust).mean()


def make_optimizer(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        rate = config.lr if name.startswith(("features.", "head_norm.")) else (config.head_lr or config.lr * config.head_lr_multiplier)
        decay = config.weight_decay if parameter.ndim > 1 else 0.0
        groups.setdefault((rate, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": parameters, "lr": rate, "weight_decay": decay}
                             for (rate, decay), parameters in groups.items()])


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source.detach(), 1 - decay)
    # BatchNorm 的整型计数器不能作指数平均。
    for target, source in zip(ema.buffers(), model.buffers()):
        if target.is_floating_point():
            target.lerp_(source.detach(), 1 - decay)
        else:
            target.copy_(source)
