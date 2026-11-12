"""东北大学机器学习实验二：完整的 PyTorch 人脸表情六分类流程。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")  # 在服务器、Kaggle 和无桌面的环境也能保存图表。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score, log_loss)
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torchvision
from torchvision import models, transforms as T

from prepare_data import CLASS_NAMES, CLASS_TO_LABEL, prepare_data, write_json

PROJECT = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    # 顶层函数可被 Windows spawn 子进程导入；不能用 lambda 代替。
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def choose_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。请安装对应的 CUDA 版 PyTorch，或传 --device cpu。")
    return torch.device(request)


def make_transform(arch: str, image_size: int, mean: list, std: list,
                   training: bool, augment: bool, augment_strength: float = 1.0,
                   geometry_strength: float | None = None,
                   photo_strength: float | None = None) -> T.Compose:
    steps = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR)]
    if training and augment and augment_strength > 0:
        # 轻度姿态和光照增强；大旋转/大裁剪可能破坏眉眼、嘴部等表情证据。
        # 1.0 精确对应原基线；只缩放扰动幅度，水平翻转概率和随机操作顺序不变。
        strength = augment_strength if geometry_strength is None else geometry_strength
        photo = augment_strength if photo_strength is None else photo_strength
        steps.extend([T.RandomHorizontalFlip(p=0.5),
                      T.RandomAffine(degrees=10 * strength,
                                     translate=(0.04 * strength, 0.04 * strength),
                                     scale=(1 - 0.08 * strength, 1 + 0.08 * strength),
                                     interpolation=T.InterpolationMode.BILINEAR, fill=128),
                      T.ColorJitter(brightness=0.15 * photo, contrast=0.15 * photo)])
    steps.extend([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return T.Compose(steps)


class EmotionDataset(Dataset):
    def __init__(self, array_path: Path, indices: np.ndarray, labels: np.ndarray | None,
                 transform: T.Compose, rgb: bool, sample_weights: np.ndarray | None = None):
        self.array_path = str(array_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.transform = transform
        self.rgb = rgb
        self.sample_weights = sample_weights
        self._images = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None  # 每个 worker 自己 mmap，避免复制整个数据数组。
        return state

    def __getitem__(self, index: int):
        if self._images is None:
            self._images = np.load(self.array_path, mmap_mode="r")
        array_index = int(self.indices[index])
        image = Image.fromarray(self._images[array_index])
        if self.rgb:
            # 单通道复制成三通道，不会创造颜色信息；用于匹配 ImageNet 模型输入。
            image = image.convert("RGB")
        target = -1 if self.labels is None else int(self.labels[array_index])
        result = self.transform(image), target
        if self.sample_weights is not None:
            return *result, float(self.sample_weights[array_index])
        return result


def make_loader(dataset: Dataset, args, shuffle: bool, seed: int,
                device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=device.type == "cuda",
                      persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


class ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
                                  nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
                                  nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(outgoing))
        self.shortcut = (nn.Identity() if stride == 1 and incoming == outgoing else
                         nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False),
                                       nn.BatchNorm2d(outgoing)))

    def forward(self, x):
        return torch.relu(self.main(x) + self.shortcut(x))


class SmallResNet(nn.Module):
    """适合 48x48 原始灰度图的离线基线，无需下载任何预训练权重。"""
    def __init__(self, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResidualBlock(32, 32), ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2), ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2), ResidualBlock(128, 128),
            ResidualBlock(128, 256, 2), ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, 6))

    def forward(self, x):
        return self.fc(self.features(x))


def create_model(arch: str, pretrained: bool, dropout: float) -> nn.Module:
    if arch == "smallresnet":
        return SmallResNet(dropout)
    # 只改变骨干深度；两种 ResNet 使用相同的六分类头与 ImageNet 输入规范。
    builders = {"resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
                "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT)}
    if arch not in builders:
        raise ValueError(f"不支持的模型：{arch}")
    builder, pretrained_weights = builders[arch]
    model = builder(weights=pretrained_weights if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 6))
    return model


def normalization(cache: Path, indices: np.ndarray, arch: str) -> tuple[list, list]:
    if arch in ("resnet18", "resnet34"):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # 从零训练的模型仅用当前训练折估计均值/标准差，防止验证集统计泄漏。
    images = np.load(cache / "train_images.npy", mmap_mode="r")[indices]
    return [float(images.mean() / 255.0)], [max(float(images.std() / 255.0), 0.01)]


def train_epoch(model, loader, optimizer, criterion, scaler, device, amp,
                mixup_alpha=0.0, ema=None, ema_warmup=False):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for batch in loader:
        images, labels = batch[:2]
        sample_weights = batch[2].to(device=device, dtype=torch.float32, non_blocking=True) if len(batch) == 3 else None
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ratio, shuffled_labels = 1.0, labels
        if mixup_alpha > 0:
            # 图像与标签同时混合；不能只混图像而仍按原标签计算损失。
            ratio = float(np.random.beta(mixup_alpha, mixup_alpha))
            order = torch.randperm(len(labels), device=device)
            images = ratio * images + (1 - ratio) * images[order]
            shuffled_labels = labels[order]
        optimizer.zero_grad(set_to_none=True)  # 每个 batch 清零，防止意外累积梯度。
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)  # 输入 logits，不要先做 softmax。
            if mixup_alpha > 0:
                loss = ratio * loss + (1 - ratio) * criterion(logits, shuffled_labels)
            if sample_weights is not None:
                # 权重按训练折的重复组大小计算，平均尺度保持为 1。
                loss = (loss * sample_weights).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失出现 NaN/Inf，请检查学习率和数据。")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None and scaler.get_scale() >= previous_scale:
            if ema_warmup:
                ema.module.load_state_dict(model.state_dict())
                ema.n_averaged.zero_()
            else:
                ema.update_parameters(model)
        count = len(labels)
        loss_sum += loss.detach().item() * count
        correct += ratio * float((logits.argmax(1) == labels).sum().item())
        if mixup_alpha > 0:
            correct += (1 - ratio) * float((logits.argmax(1) == shuffled_labels).sum().item())
        total += count
    return loss_sum / total, correct / total


def optimizer_groups(model, args):
    head_ids = {id(p) for p in model.fc.parameters()} if args.arch in ("resnet18", "resnet34") and not args.no_pretrained else set()
    if not args.no_norm_bias_decay:
        if head_ids:
            return [{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
                    {"params": model.fc.parameters(), "lr": args.lr * 3}]
        return model.parameters()
    norm_ids = set()
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm)):
            norm_ids.update(id(p) for p in module.parameters(recurse=False))
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = args.lr * (3 if id(parameter) in head_ids else 1)
        decay = 0.0 if id(parameter) in norm_ids or name.endswith(".bias") else args.weight_decay
        groups.setdefault((lr, decay), []).append(parameter)
    return [{"params": parameters, "lr": lr, "weight_decay": decay}
            for (lr, decay), parameters in groups.items()]


def capture_rng(train_loader, val_loader, device):
    np_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "train_generator": train_loader.generator.get_state(),
            "val_generator": val_loader.generator.get_state() if val_loader else None}


def restore_rng(state, train_loader, val_loader, device):
    random.setstate(state["python"])
    item = state["numpy"]
    np.random.set_state((item[0], np.asarray(item[1], dtype=np.uint32), item[2], item[3], item[4]))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    train_loader.generator.set_state(state["train_generator"].cpu())
    if val_loader is not None and state["val_generator"] is not None:
        val_loader.generator.set_state(state["val_generator"].cpu())


@torch.inference_mode()
def predict_probabilities(model, loader, device, amp: bool, tta: bool):
    model.eval()  # 关闭 Dropout，并使用 BatchNorm 的运行统计。
    probabilities, raw_probabilities = [], []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            raw = model(images).float().softmax(1)
            prediction = raw
            if tta:
                flipped = model(torch.flip(images, dims=[3])).float().softmax(1)
                prediction = 0.5 * (raw + flipped)
        probabilities.append(prediction.cpu().numpy())
        raw_probabilities.append(raw.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(raw_probabilities)


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = probabilities.argmax(1)
    return {"accuracy": float(accuracy_score(labels, prediction)),
            "macro_f1": float(f1_score(labels, prediction, labels=list(range(6)),
                                       average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            "log_loss": float(log_loss(labels, probabilities, labels=list(range(6))))}


def plot_data(cache: Path, output: Path) -> None:
    manifest = pd.read_csv(cache / "train_manifest.csv")
    images = np.load(cache / "train_images.npy", mmap_mode="r")
    counts = manifest.groupby("class_name").size().reindex(CLASS_NAMES)
    figure, axis = plt.subplots(figsize=(9, 4))
    bars = axis.bar(CLASS_NAMES, counts.values, color="#3475ae")
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Number of clean training images")
    axis.set_ylim(0, int(counts.max() * 1.15))
    figure.tight_layout()
    figure.savefig(output / "class_distribution.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(6, 5, figsize=(8, 10))
    for label, name in enumerate(CLASS_NAMES):
        example_indices = manifest.index[manifest.label == label][:5]
        for column, index in enumerate(example_indices):
            axes[label, column].imshow(images[index], cmap="gray", vmin=0, vmax=255)
            axes[label, column].axis("off")
            axes[label, column].set_title(name if column == 0 else "", fontsize=9)
    figure.tight_layout()
    figure.savefig(output / "sample_images.png", dpi=150)
    plt.close(figure)


def plot_history(history: list[dict], output: Path) -> None:
    table = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(table.epoch, table.train_loss, label="Train (smoothed CE)")
    if "val_log_loss" in table:
        axes[0].plot(table.epoch, table.val_log_loss, label="Validation (TTA/plain CE)")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(table.epoch, table.train_accuracy, label="Train (augmented)")
    if "val_accuracy" in table:
        axes[1].plot(table.epoch, table.val_accuracy, label="Validation (TTA)")
        axes[1].plot(table.epoch, table.val_raw_accuracy, label="Validation (no TTA)", alpha=0.7)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=150)
    plt.close(figure)


def save_validation(manifest, indices, probabilities, raw, cache, output) -> dict:
    labels = manifest.label.to_numpy()[indices]
    metrics = calculate_metrics(labels, probabilities)
    metrics["accuracy_without_tta"] = float(accuracy_score(labels, raw.argmax(1)))
    metrics["validation_count"] = len(indices)
    report = classification_report(labels, probabilities.argmax(1), labels=list(range(6)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    write_json(output / "validation_metrics.json", metrics)
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
    predictions = manifest.iloc[indices][["path", "ID", "label", "pixel_hash"]].copy()
    predictions["prediction"] = probabilities.argmax(1)
    predictions["confidence"] = probabilities.max(1)
    for label, name in enumerate(CLASS_NAMES):
        predictions[f"p_{name}"] = probabilities[:, label]
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    matrix = confusion_matrix(labels, probabilities.argmax(1), labels=list(range(6)))
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output / "confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Validation confusion matrix")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    errors = np.flatnonzero(labels != probabilities.argmax(1))
    if len(errors):
        selected = errors[np.argsort(-probabilities[errors].max(1))[:12]]
        images = np.load(cache / "train_images.npy", mmap_mode="r")
        figure, axes = plt.subplots(3, 4, figsize=(11, 8))
        for axis in axes.flat:
            axis.axis("off")
        for axis, relative_index in zip(axes.flat, selected):
            axis.imshow(images[indices[relative_index]], cmap="gray", vmin=0, vmax=255)
            axis.set_title(f"True: {CLASS_NAMES[labels[relative_index]]}\n"
                           f"Pred: {CLASS_NAMES[probabilities[relative_index].argmax()]} "
                           f"({probabilities[relative_index].max():.2f})", fontsize=9)
        figure.tight_layout()
        figure.savefig(output / "confident_errors.png", dpi=150)
        plt.close(figure)
    return metrics


def save_submission(cache: Path, probabilities: np.ndarray, output: Path) -> dict:
    sample = pd.read_csv(cache / "sample_submission.csv", dtype={"ID": str})
    if probabilities.shape != (len(sample), 6) or not np.isfinite(probabilities).all():
        raise ValueError("测试概率的维度或有限性检查失败。")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("概率不是合法的六类分布。")
    submission = sample.copy()
    submission["Emotion"] = probabilities.argmax(1).astype(np.int64)
    csv_path = output / "submission.csv"
    submission.to_csv(csv_path, index=False)
    # 回读真正落盘的文件，检查表头、行数、ID 顺序、缺失值与整数标签。
    reloaded = pd.read_csv(csv_path, dtype={"ID": str})
    assert reloaded.columns.tolist() == ["ID", "Emotion"]
    assert len(reloaded) == len(sample)
    assert reloaded.ID.tolist() == sample.ID.tolist()
    assert not reloaded.isna().any().any() and not reloaded.ID.duplicated().any()
    assert pd.api.types.is_integer_dtype(reloaded.Emotion)
    assert reloaded.Emotion.between(0, 5).all()
    np.save(output / "test_probabilities.npy", probabilities)
    check = {"rows": len(reloaded), "columns": reloaded.columns.tolist(),
             "id_order_matches_template": True, "unique_ids": True,
             "no_missing_values": True, "integer_labels_in_0_5": True,
             "prediction_counts": {name: int((reloaded.Emotion == label).sum())
                                   for name, label in CLASS_TO_LABEL.items()},
             "competition_submission_performed": False}
    write_json(output / "submission_checks.json", check)
    return check


def environment_info(device: torch.device) -> dict:
    return {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__, "pillow": Image.__version__,
            "device": str(device), "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def fit_model(args, cache, manifest, train_indices, val_indices, fold_name, device, audit):
    fold_output = Path(args.output) / fold_name
    fold_output.mkdir(parents=True, exist_ok=True)
    train_seed = args.seed if args.train_seed is None else args.train_seed
    seed = train_seed + (int(fold_name.split("_")[-1]) if fold_name.startswith("fold_") else 100)
    seed_everything(seed)
    mean, std = normalization(cache, train_indices, args.arch)
    config = vars(args).copy()
    config.update({"mean": mean, "std": std, "class_to_label": CLASS_TO_LABEL,
                   "source_fingerprint": audit["fingerprint"], "fold_name": fold_name,
                   "train_count": len(train_indices), "val_count": len(val_indices),
                   "effective_train_seed": seed,
                   "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_json(fold_output / "config.json", config)
    labels = manifest.label.to_numpy()
    sample_weights = None
    if args.duplicate_weighting:
        counts = manifest.iloc[train_indices].groupby("pixel_hash").size()
        sample_weights = np.ones(len(manifest), dtype=np.float32)
        sample_weights[train_indices] = (1 / manifest.iloc[train_indices].pixel_hash.map(counts)).to_numpy()
        sample_weights[train_indices] /= sample_weights[train_indices].mean()
        write_json(fold_output / "duplicate_weights.json", {
            "weighted_samples": int((manifest.iloc[train_indices].pixel_hash.map(counts) > 1).sum()),
            "unique_groups": len(counts), "mean_training_weight": float(sample_weights[train_indices].mean()),
            "validation_weights_changed": False})
    training_data = EmotionDataset(cache / "train_images.npy", train_indices, labels,
                                  make_transform(args.arch, args.image_size, mean, std,
                                                 True, not args.no_augment, args.augment_strength,
                                                 args.geometry_strength, args.photo_strength),
                                  args.arch in ("resnet18", "resnet34"), sample_weights)
    train_loader = make_loader(training_data, args, True, seed, device)
    val_loader = None
    if len(val_indices):
        validation_data = EmotionDataset(cache / "train_images.npy", val_indices, labels,
                                        make_transform(args.arch, args.image_size, mean, std,
                                                       False, False), args.arch in ("resnet18", "resnet34"))
        val_loader = make_loader(validation_data, args, False, seed, device)

    model = create_model(args.arch, not args.no_pretrained, args.dropout).to(device)
    groups = optimizer_groups(model, args)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    ema = None
    if args.ema_decay > 0:
        # 同时平均浮点参数和 BN 缓冲区，整数计数缓冲由 PyTorch 正确处理。
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)
        ema.requires_grad_(False)

    def lr_factor(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs - 1, 1)
        return 0.02 + 0.98 * (1 + math.cos(math.pi * min(progress, 1))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing,
                                   reduction="none" if sample_weights is not None else "mean")
    best_accuracy, best_loss, stale, best_epoch, history = -1.0, float("inf"), 0, 0, []
    best_raw_accuracy, best_raw_loss, start_epoch = -1.0, float("inf"), 1
    last_path = fold_output / "last.pt"
    if args.resume and last_path.exists():
        restored = torch.load(last_path, map_location=device, weights_only=True)
        for key in ["source_fingerprint", "arch", "image_size", "effective_train_seed", "epochs",
                    "lr", "weight_decay", "dropout", "label_smoothing", "augment_strength",
                    "geometry_strength", "photo_strength", "ema_decay", "ema_warmup_epochs",
                    "mixup_alpha", "no_norm_bias_decay", "duplicate_weighting", "batch_size"]:
            if restored["config"][key] != config[key]:
                raise ValueError(f"恢复训练配置改变：{key}")
        model.load_state_dict(restored["model_state"])
        optimizer.load_state_dict(restored["optimizer_state"])
        scheduler.load_state_dict(restored["scheduler_state"])
        scaler.load_state_dict(restored["scaler_state"])
        if ema is not None:
            ema.load_state_dict(restored["ema_state"])
        history = restored["history"]
        best_accuracy, best_loss = restored["best_accuracy"], restored["best_loss"]
        best_epoch, stale = restored["best_epoch"], restored["stale"]
        best_raw_accuracy, best_raw_loss = restored["best_raw_accuracy"], restored["best_raw_loss"]
        start_epoch = restored["epoch"] + 1
        restore_rng(restored["rng"], train_loader, val_loader, device)
        print(f"恢复 {fold_name}，从第 {start_epoch} 轮继续。", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, scaler, device, amp,
                                                 args.mixup_alpha, ema, epoch <= args.ema_warmup_epochs)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
               "lr": optimizer.param_groups[0]["lr"]}
        improved = val_loader is None
        if val_loader is not None:
            evaluated = ema.module if ema is not None else model
            probabilities, raw = predict_probabilities(evaluated, val_loader, device, amp, not args.no_tta)
            metrics = calculate_metrics(labels[val_indices], probabilities)
            row.update({"val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
                        "val_log_loss": metrics["log_loss"],
                        "val_raw_accuracy": float(accuracy_score(labels[val_indices], raw.argmax(1)))})
            if ema is not None:
                ordinary, _ = predict_probabilities(model, val_loader, device, amp, not args.no_tta)
                ordinary_metrics = calculate_metrics(labels[val_indices], ordinary)
                row.update({"ordinary_val_accuracy": ordinary_metrics["accuracy"],
                            "ordinary_val_macro_f1": ordinary_metrics["macro_f1"],
                            "ordinary_val_log_loss": ordinary_metrics["log_loss"]})
                if (ordinary_metrics["accuracy"] > best_raw_accuracy or
                        (ordinary_metrics["accuracy"] == best_raw_accuracy and ordinary_metrics["log_loss"] < best_raw_loss)):
                    best_raw_accuracy, best_raw_loss = ordinary_metrics["accuracy"], ordinary_metrics["log_loss"]
                    torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch,
                                "validation_accuracy": best_raw_accuracy, "evaluation_model": "ordinary"},
                               fold_output / "best_raw.pt")
            improved = (metrics["accuracy"] > best_accuracy or
                        (metrics["accuracy"] == best_accuracy and metrics["log_loss"] < best_loss))
            if improved:
                best_accuracy, best_loss = metrics["accuracy"], metrics["log_loss"]
        scheduler.step()
        row["epoch_seconds"] = time.perf_counter() - started
        history.append(row)
        pd.DataFrame(history).to_csv(fold_output / "history.csv", index=False)
        if improved:
            best_epoch, stale = epoch, 0
            selected_model = ema.module if ema is not None else model
            torch.save({"model_state": selected_model.state_dict(), "config": config,
                        "epoch": epoch, "validation_accuracy": best_accuracy if val_loader else None,
                        "evaluation_model": "ema" if ema is not None else "ordinary"},
                       fold_output / "best.pt")
        else:
            stale += 1
        # last.pt 保存训练状态；best.pt 只保存用于推理的最佳模型。
        temporary_last = fold_output / "last.tmp"
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                    "ema_state": ema.state_dict() if ema is not None else None, "config": config,
                    "epoch": epoch, "history": history, "best_accuracy": best_accuracy, "best_loss": best_loss,
                    "best_epoch": best_epoch, "stale": stale, "best_raw_accuracy": best_raw_accuracy,
                    "best_raw_loss": best_raw_loss, "rng": capture_rng(train_loader, val_loader, device)}, temporary_last)
        temporary_last.replace(last_path)
        message = f"{fold_name} epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}"
        if val_loader is not None:
            message += f", val_acc={row['val_accuracy']:.4f}, val_macro_f1={row['val_macro_f1']:.4f}"
        print(message + f", seconds={row['epoch_seconds']:.1f}", flush=True)
        if val_loader is not None and args.patience > 0 and stale >= args.patience:
            print(f"连续 {args.patience} 轮未改善，提前停止。", flush=True)
            break
    plot_history(history, fold_output)
    saved = torch.load(fold_output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    result = {"fold": fold_name, "best_epoch": best_epoch,
              "epochs_completed": len(history), "checkpoint": str(fold_output / "best.pt"),
              "training_seconds": sum(h["epoch_seconds"] for h in history)}
    val_probabilities = None
    if val_loader is not None:
        val_probabilities, raw = predict_probabilities(model, val_loader, device, amp, not args.no_tta)
        result.update(save_validation(manifest, val_indices, val_probabilities, raw, cache, fold_output))
    else:
        result["evaluation_note"] = "全量重训没有独立验证集，不能报告验证准确率。"
    if ema is not None and val_loader is not None:
        ordinary_saved = torch.load(fold_output / "best_raw.pt", map_location=device, weights_only=True)
        model.load_state_dict(ordinary_saved["model_state"])
        ordinary_probabilities, ordinary_raw = predict_probabilities(model, val_loader, device, amp, not args.no_tta)
        ordinary_output = fold_output / "ordinary_model"
        ordinary_output.mkdir(exist_ok=True)
        result["ordinary_model_metrics"] = save_validation(manifest, val_indices, ordinary_probabilities,
                                                            ordinary_raw, cache, ordinary_output)
        model.load_state_dict(saved["model_state"])
    test_loader, test_probabilities = None, None
    if not args.validation_only:
        test_data = EmotionDataset(cache / "test_images.npy", np.arange(audit["test_count"]), None,
                                   make_transform(args.arch, args.image_size, mean, std, False, False),
                                   args.arch in ("resnet18", "resnet34"))
        test_loader = make_loader(test_data, args, False, seed, device)
        test_probabilities, _ = predict_probabilities(model, test_loader, device, amp, not args.no_tta)
        save_submission(cache, test_probabilities, fold_output)
    write_json(fold_output / "run_summary.json", result)
    del model, ema, optimizer, scaler, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, val_probabilities, test_probabilities


def train(args) -> dict:
    output, cache = Path(args.output).resolve(), Path(args.cache).resolve()
    args.output = str(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    torch.hub.set_dir(str(Path(args.weights_dir).resolve()))
    device = choose_device(args.device)
    audit = prepare_data(args.data, cache)
    print(f"数据审计：训练 {audit['clean_train_count']}，测试 {audit['test_count']}；设备 {device}", flush=True)
    write_json(output / "environment.json", environment_info(device))
    write_json(output / "data_audit.json", audit)
    plot_data(cache, output)
    manifest = pd.read_csv(cache / "train_manifest.csv")
    labels, hashes = manifest.label.to_numpy(), manifest.pixel_hash.to_numpy()
    if args.full_data:
        splits = [("refit", np.arange(len(manifest)), np.array([], dtype=np.int64))]
        complete_cv = False
    else:
        split_seed = args.seed if args.split_seed is None else args.split_seed
        if args.split_manifest:
            fixed = pd.read_csv(args.split_manifest)
            for key in ["path", "ID", "label", "pixel_hash"]:
                if fixed[key].tolist() != manifest[key].tolist():
                    raise ValueError(f"固定划分清单与数据不一致：{key}")
            if set(fixed.validation_fold) != set(range(args.n_splits)):
                raise ValueError("固定清单的折号与 n_splits 不一致。")
            all_splits = [(np.flatnonzero(fixed.validation_fold.to_numpy() != fold),
                           np.flatnonzero(fixed.validation_fold.to_numpy() == fold))
                          for fold in range(args.n_splits)]
        else:
            splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=split_seed)
            all_splits = list(splitter.split(np.zeros(len(manifest)), labels, hashes))
        requested = list(range(args.n_splits)) if args.folds == "all" else [int(i) for i in args.folds.split(",")]
        if len(set(requested)) != len(requested) or any(i < 0 or i >= args.n_splits for i in requested):
            raise ValueError("--folds 必须是 all 或不重复且位于 0..n_splits-1 的编号列表。")
        split_table = manifest[["path", "ID", "label", "pixel_hash"]].copy()
        split_table["validation_fold"] = -1
        for fold, (training, validation) in enumerate(all_splits):
            if set(hashes[training]) & set(hashes[validation]):
                raise AssertionError("同一像素哈希同时出现在训练和验证集。")
            if set(labels[training]) != set(range(6)) or set(labels[validation]) != set(range(6)):
                raise ValueError("某一折缺少类别，请减少折数。")
            split_table.loc[validation, "validation_fold"] = fold
        split_table.to_csv(output / "split_manifest.csv", index=False)
        write_json(output / "split_checks.json", {
            "n_splits": args.n_splits, "seed": split_seed, "pixel_hash_overlap_between_train_val": 0,
            "fold_sizes": [len(v) for _, v in all_splits],
            "fold_class_counts": [{name: int((labels[v] == label).sum())
                                   for name, label in CLASS_TO_LABEL.items()} for _, v in all_splits]})
        splits = [(f"fold_{i}", *all_splits[i]) for i in requested]
        complete_cv = len(requested) == args.n_splits

    results, test_predictions = [], []
    oof = np.full((len(manifest), 6), np.nan, dtype=np.float32)
    for name, training, validation in splits:
        print(f"开始 {name}：训练 {len(training)}，验证 {len(validation)}", flush=True)
        result, val_probabilities, test_probabilities = fit_model(
            args, cache, manifest, training, validation, name, device, audit)
        results.append(result)
        if test_probabilities is not None:
            test_predictions.append(test_probabilities)
        if len(validation):
            oof[validation] = val_probabilities
    checks = None
    if test_predictions:
        ensemble = np.mean(test_predictions, axis=0)
        checks = save_submission(cache, ensemble, output)
    summary = {"runs": results, "complete_cross_validation": complete_cv,
               "data_fingerprint": audit["fingerprint"], "submission_checks": checks,
               "test_prediction_performed": bool(test_predictions)}
    valid = np.isfinite(oof).all(axis=1)
    if valid.any():
        summary["pooled_validation_metrics"] = calculate_metrics(labels[valid], oof[valid])
        summary["evaluated_validation_count"] = int(valid.sum())
        np.save(output / "oof_probabilities.npy", oof)
    write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def predict(args) -> dict:
    cache, output = Path(args.cache).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    audit = prepare_data(args.data, cache)
    device = choose_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    predictions = []
    for checkpoint in args.checkpoints:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        config = saved["config"]
        if config["class_to_label"] != CLASS_TO_LABEL:
            raise ValueError("checkpoint 类别编号与题目不一致。")
        model = create_model(config["arch"], False, config["dropout"]).to(device)
        model.load_state_dict(saved["model_state"])
        data = EmotionDataset(cache / "test_images.npy", np.arange(audit["test_count"]), None,
                              make_transform(config["arch"], config["image_size"],
                                             config["mean"], config["std"], False, False),
                              config["arch"] in ("resnet18", "resnet34"))
        loader = make_loader(data, args, False, args.seed, device)
        probabilities, _ = predict_probabilities(model, loader, device,
                                                  device.type == "cuda" and not args.no_amp,
                                                  not args.no_tta)
        predictions.append(probabilities)
        del model, loader
    checks = save_submission(cache, np.mean(predictions, axis=0), output)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="仅数据审计，不训练")
    audit.add_argument("--data", required=True)
    audit.add_argument("--cache", default=str(PROJECT / "data_cache"))
    audit.add_argument("--refresh", action="store_true")
    for command in ["train", "predict"]:
        child = commands.add_parser(command)
        child.add_argument("--data", required=True)
        child.add_argument("--cache", default=str(PROJECT / "data_cache"))
        child.add_argument("--output", default=str(PROJECT / "outputs"))
        child.add_argument("--batch-size", type=int, default=128)
        child.add_argument("--workers", type=int, default=2)
        child.add_argument("--cpu-threads", type=int, default=6)
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        child.add_argument("--no-amp", action="store_true")
        child.add_argument("--no-tta", action="store_true")
        if command == "predict":
            child.add_argument("--checkpoints", nargs="+", required=True)
        else:
            child.add_argument("--arch", choices=["resnet18", "resnet34", "smallresnet"], default="resnet18")
            child.add_argument("--no-pretrained", action="store_true")
            child.add_argument("--image-size", type=int, default=96)
            child.add_argument("--epochs", type=int, default=25)
            child.add_argument("--patience", type=int, default=8)
            child.add_argument("--warmup-epochs", type=int, default=2)
            child.add_argument("--lr", type=float, default=5e-4)
            child.add_argument("--weight-decay", type=float, default=1e-4)
            child.add_argument("--label-smoothing", type=float, default=0.05)
            child.add_argument("--dropout", type=float, default=0.2)
            child.add_argument("--no-augment", action="store_true")
            child.add_argument("--augment-strength", type=float, default=1.0,
                               help="0 关闭增强，0.5 弱增强，1 原基线，1.5 强增强")
            child.add_argument("--geometry-strength", type=float, default=None,
                               help="单独控制旋转、平移和缩放；默认沿用 augment-strength")
            child.add_argument("--photo-strength", type=float, default=None,
                               help="单独控制亮度与对比度；默认沿用 augment-strength")
            child.add_argument("--split-seed", type=int, default=None)
            child.add_argument("--train-seed", type=int, default=None)
            child.add_argument("--split-manifest", default=None, help="复用固定划分清单")
            child.add_argument("--ema-decay", type=float, default=0.0, help="0 关闭 EMA；试验使用 0.99")
            child.add_argument("--ema-warmup-epochs", type=int, default=2)
            child.add_argument("--no-norm-bias-decay", action="store_true")
            child.add_argument("--mixup-alpha", type=float, default=0.0)
            child.add_argument("--duplicate-weighting", action="store_true")
            child.add_argument("--resume", action="store_true", help="从各折 last.pt 恢复优化器等状态")
            child.add_argument("--weights-dir", default=str(PROJECT / "weights_cache"))
            child.add_argument("--validation-only", action="store_true",
                               help="调参只生成验证产物，选中权重后再预测测试集")
            child.add_argument("--n-splits", type=int, default=5)
            child.add_argument("--folds", default="0", help="例如 0、0,1 或 all")
            child.add_argument("--full-data", action="store_true", help="按确定的轮数全量重训，不报告验证分数")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        result = prepare_data(args.data, args.cache, args.refresh)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if args.batch_size < 1 or args.workers < 0 or args.cpu_threads < 1:
        raise ValueError("batch-size、cpu-threads 必须为正，workers 不能为负。")
    if args.command == "train":
        if args.epochs < 1 or args.n_splits < 2 or args.image_size < 16:
            raise ValueError("epochs >= 1，n-splits >= 2，image-size >= 16。")
        if args.lr <= 0 or not 0 <= args.label_smoothing < 1 or not 0 <= args.dropout < 1:
            raise ValueError("学习率必须为正，label-smoothing 和 dropout 必须位于 [0,1)。")
        if args.weight_decay < 0 or args.warmup_epochs < 0:
            raise ValueError("weight-decay 和 warmup-epochs 不能为负。")
        if not math.isfinite(args.augment_strength) or not 0 <= args.augment_strength <= 3:
            raise ValueError("augment-strength 必须为 0～3 的有限数。")
        for value in [args.geometry_strength, args.photo_strength]:
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 3):
                raise ValueError("分项增强强度必须为 0～3 的有限数。")
        if not 0 <= args.ema_decay < 1 or args.ema_warmup_epochs < 0:
            raise ValueError("EMA 衰减必须位于 [0,1)，预热轮数必须非负。")
        if not math.isfinite(args.mixup_alpha) or args.mixup_alpha < 0:
            raise ValueError("mixup-alpha 必须是非负有限数。")
        if args.mixup_alpha > 0 and args.duplicate_weighting:
            raise ValueError("本轮分开比较 MixUp 与重复降权，不能同时启用。")
        if args.validation_only and args.full_data:
            raise ValueError("validation-only 需要验证集，不能与 full-data 同时使用。")
        if args.arch == "smallresnet":
            args.no_pretrained = True
        return train(args)
    return predict(args)


if __name__ == "__main__":
    main()
