"""表情领域预训练 CNN：严格加载作者权重、保留原注意力、正确迁移六分类头。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model_and_aug import Config as BaseConfig

ROOT = Path(__file__).resolve().parent
if (ROOT/"vendor_timm_modern").is_dir():
    # Notebook 的 sys.path 不会被新 Python 子进程继承；每折主动加载固定版本。
    sys.path.insert(0,str(ROOT/"vendor_timm_modern"))
ARCHITECTURES = {"emotieffnet_b2", "ddamfnpp_affectnet7", "dan_affectnet7"}
LABEL_ROWS = {"emotieffnet_b2": [0,2,3,5,6,4],
              "ddamfnpp_affectnet7": [6,4,1,2,3,0], "dan_affectnet7": [6,4,1,2,3,0]}
NATIVE_SIZES = {"emotieffnet_b2":260, "ddamfnpp_affectnet7":112, "dan_affectnet7":224}


@dataclass
class Config(BaseConfig):
    architecture: str = "emotieffnet_b2"
    head: str = "gap"
    loss_kind: str = "ce"
    gce_q: float = 0.7
    gce_weight: float = 0.5
    robust_start_epoch: int = 5
    selection: str = "ema"
    clean_train_samples: int = 600
    head_lr: float | None = 1e-4
    freeze_bn: bool = True
    freeze_stages: int = 0

    def validate(self):
        super().validate()
        assert self.architecture in ARCHITECTURES and self.head == "gap"
        assert self.attention == "none", "保留作者已训练注意力；本轮不叠加随机注意力"
        assert self.architecture != "ddamfnpp_affectnet7" or self.image_size == 112
        assert self.loss_kind in {"ce", "ce_gce", "focal"}
        assert 0 < self.gce_q <= 1 and 0 <= self.gce_weight <= 1
        assert self.selection in {"ordinary", "ema"}
        assert self.selection != "ema" or self.ema_decay > 0
        assert self.freeze_stages >= 0 and self.clean_train_samples >= 6
        assert self.head_lr is None or self.head_lr > 0
        return self


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_checkpoint(path):
    # 部分作者将 numpy 标量验证指标一起保存；只允许这些数值类型，不放开任意 pickle。
    allowed = {"numpy._core.multiarray.scalar":np._core.multiarray.scalar,
               "numpy.core.multiarray.scalar":np._core.multiarray.scalar,
               "numpy.dtype":np.dtype}
    symbols = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    assert symbols <= set(allowed), f"权重包含未审阅对象：{symbols-set(allowed)}"
    globals_list = [(allowed[name], name) for name in symbols]
    globals_list += [type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(globals_list):
        return torch.load(path, map_location="cpu", weights_only=True)


def ensure_weights(architecture, directory, existing=None):
    assert architecture in ARCHITECTURES
    directory = Path(directory)
    names = {"emotieffnet_b2":"emotieffnet_b2_affectnet7.safetensors",
             "ddamfnpp_affectnet7":"ddamfnpp_affectnet7.pth", "dan_affectnet7":"dan_affectnet7.pth"}
    candidate = Path(existing) if existing else directory / names[architecture]
    if not candidate.is_file():
        raise FileNotFoundError(f"缺少官方权重：{candidate}；不使用随机权重代替")
    if architecture == "emotieffnet_b2":
        conversion = json.loads((candidate.parent / "emotieffnet_b2_conversion.json").read_text())
        assert conversion["passed"] and file_hash(candidate) == conversion["converted_sha256"]
    else:
        public = json.loads((directory / "public_weight_manifest.json").read_text(encoding="utf-8"))
        assert file_hash(candidate) == public[architecture]["sha256"]
    return candidate


def native_model(architecture, weights=None):
    if architecture == "emotieffnet_b2":
        import timm
        assert timm.__version__ == "1.0.29", "训练进程必须使用已核验的现代 timm 版本"
        from safetensors.torch import load_file
        # 作者采用 max pooling；默认 EfficientNet 的 avg pooling 会改变表情特征。
        model = timm.create_model("tf_efficientnet_b2", pretrained=False, num_classes=7, global_pool="max")
        if weights is not None:
            model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    else:
        if architecture == "ddamfnpp_affectnet7":
            from fer_nets.ddamfn_pp import DDAMNet
            model = DDAMNet(num_class=7, num_head=2, pretrained=False)
        else:
            from fer_nets.dan import DAN
            model = DAN(num_class=7, num_head=4, pretrained=False)
        if weights is not None:
            saved = safe_checkpoint(weights)
            assert isinstance(saved, dict) and "model_state_dict" in saved
            state = saved["model_state_dict"]
            # DataParallel 只改变键名前缀；严格加载仍检查全部参数和缓冲区。
            if all(key.startswith("module.") for key in state):
                state = {key[7:]:value for key,value in state.items()}
            model.load_state_dict(state, strict=True)
    return model


class FeatureWrapper(nn.Module):
    def __init__(self, body, architecture):
        super().__init__()
        self.backbone = body
        self.architecture = architecture

    def forward(self, images):
        # DDAMFN/DAN 的原注意力仍在 backbone 内；fc 已改为 Identity。
        value = self.backbone(images)
        if isinstance(value, tuple):
            value = value[0]
        return value[:, :, None, None]


class CNN(nn.Module):
    def __init__(self, config, weights=None):
        super().__init__()
        config.validate()
        self.config = config
        self.native_attention = True  # EfficientNet 自带 SE；另两者保留作者训练好的区域注意力。
        body = native_model(config.architecture, weights)
        old = body.classifier if config.architecture == "emotieffnet_b2" else body.fc
        channels = old.in_features
        self.classifier = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(channels, 6))
        rows = torch.tensor(LABEL_ROWS[config.architecture])
        # 复制对应情感的预训练分类参数，保留表达知识；移除厌恶并按本题重排。
        with torch.no_grad():
            self.classifier[-1].weight.copy_(old.weight[rows])
            self.classifier[-1].bias.copy_(old.bias[rows])
        self.output_norm = nn.Identity()
        if config.architecture == "dan_affectnet7":
            self.output_norm = nn.BatchNorm1d(6)
            with torch.no_grad():
                for attr in ["weight","bias","running_mean","running_var"]:
                    getattr(self.output_norm,attr).copy_(getattr(body.bn,attr)[rows])
                self.output_norm.num_batches_tracked.copy_(body.bn.num_batches_tracked)
        if config.architecture == "emotieffnet_b2":
            body.classifier = nn.Identity()
        else:
            body.fc = nn.Identity()
            body.bn = nn.Identity()
        self.features = FeatureWrapper(body, config.architecture)
        if config.freeze_stages:
            early = ([body.conv_stem, body.bn1] + list(body.blocks.children())[:config.freeze_stages]
                     if config.architecture == "emotieffnet_b2" else list(body.features.children())[:config.freeze_stages])
            for module in early:
                module.requires_grad_(False)

    def forward(self, images):
        features = self.features(images).flatten(1)
        return self.output_norm(self.classifier(features))

    def train(self, mode=True):
        super().train(mode)
        # 小域迁移时先固定 BN 运行统计；gamma/beta 可继续学习，避免灰度批次污染统计。
        if mode and self.config.freeze_bn:
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
        return self


def single_loss(logits, labels, config, epoch):
    logits = logits.float()
    ce = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing, reduction="none")
    if config.loss_kind == "ce" or (config.loss_kind == "ce_gce" and epoch <= config.robust_start_epoch):
        return ce.mean()
    selected = logits.softmax(1).gather(1, labels[:,None]).squeeze(1).clamp_min(1e-7)
    if config.loss_kind == "focal":
        return ((1-selected).pow(1.5)*ce).mean()
    robust = (1-selected.pow(config.gce_q))/config.gce_q
    return ((1-config.gce_weight)*ce+config.gce_weight*robust).mean()


def make_optimizer(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = "backbone" if name.startswith("features.") else "head"
        rate = config.lr if kind == "backbone" else (config.head_lr or config.lr*config.head_lr_multiplier)
        decay = config.weight_decay if parameter.ndim > 1 else 0.0
        groups.setdefault((kind,rate,decay), []).append(parameter)
    return torch.optim.AdamW([{"params":parameters,"lr":rate,"weight_decay":decay,"group_kind":kind}
                             for (kind,rate,decay),parameters in groups.items()])


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(),model.parameters()):
        target.lerp_(source.detach(),1-decay)
    for target, source in zip(ema.buffers(),model.buffers()):
        if target.is_floating_point():
            target.lerp_(source.detach(),1-decay)
        else:
            target.copy_(source)
