"""植物幼苗分类：原版监督训练配方与精确输入缓存。

ConvNeXt-Small / 448 / batch12 / 40轮 / 五折 / seed42。
训练使用原始RGB图片；保留随机尺度裁剪、RandAugment、翻转旋转、
颜色扰动、Mixup/CutMix、标签平滑、分组学习率及EMA选模。
优化仅缓存原图解码和确定性验证张量，随机训练增强仍逐次执行。
旧JPEG预处理函数为兼容原主流程保留，PlantDS实际读取元组第一项原图。
Notebook启动器固定预训练权重并覆盖模型工厂，权重错误时明确报错。
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 0. 基础配置
# ============================================================

CFG = dict(
    # 输入：实际使用原图；以下旧预处理参数仅影响兼容JPEG缓存。
    img_size=448,
    cache_size=512,
    use_bg_removal=True,
    bg_darken=0.35,
    use_plant_crop=False,
    plant_ratio=0.55,
    crop_min_frac=0.30,
    crop_max_frac=1.00,
    aug_level="strong",
    # ImageNet预训练ConvNeXt-Small。
    arch="convnext_small.fb_in22k_ft_in1k",
    drop_rate=0.1,
    drop_path_rate=0.1,
    # 每折400张训练、100张验证；保留原版所有训练超参数。
    n_folds=5,
    epochs=40,
    batch_size=12,
    lr=3e-4,                 # 分类头3e-4，backbone为其0.3倍，即9e-5。
    weight_decay=0.05,
    warmup_epochs=3,
    label_smoothing=0.10,
    mixup_alpha=0.2,
    cutmix_alpha=1.0,
    mix_prob=0.3,
    ema_decay=0.999,
    grad_clip=1.0,
    tta=False,
    seed=42,
    num_workers=2,
    out_dir="/kaggle/working" if os.path.isdir("/kaggle/working") else "./plant_out",
)


# ============================================================
# 1. 工具函数
# ============================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def find_dirs(root_hints=None):
    """自动定位 train / test 目录(Kaggle 挂载路径不一定相同)"""
    roots = []
    for h in (root_hints or []):
        if h and os.path.isdir(h):
            roots.append(h)
    for base in ["/kaggle/input", "/kaggle/input/competitions", "/home/ubuntu/2026task1", "."]:
        if os.path.isdir(base):
            roots.append(base)

    train_dir = test_dir = None
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath[len(root):].count(os.sep) > 6:   # 限制深度,避免遍历整个 /kaggle/input
                dirnames[:] = []
                continue
            name = os.path.basename(dirpath).lower()
            if train_dir is None and name == "train":
                subs = [d for d in dirnames if not d.startswith(".")]
                if len(subs) >= 2:      # train/ 下有多个类别子目录
                    train_dir = dirpath
            if test_dir is None and name == "test":
                imgs = [f for f in filenames if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if len(imgs) >= 10:     # test/ 下直接是图片文件
                    test_dir = dirpath
        if train_dir and test_dir:
            break
    if not train_dir or not test_dir:
        raise FileNotFoundError(
            f"未能定位 train/test 目录。已搜索: {roots}\n"
            "请把数据集挂载到 /kaggle/input 下,或用 root_hints 指定路径。"
        )
    return train_dir, test_dir


def read_course_data(root, classes):
    """Index the official layout and use only the template's ID column."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    candidates = [
        train for train in root.rglob("train")
        if train.is_dir() and train.with_name("test").is_dir()
        and sorted(child.name for child in train.iterdir() if child.is_dir()) == classes
    ]
    if len(candidates) != 1:
        raise ValueError("Expected one train/test pair with the five official class folders")
    train = candidates[0]
    test = train.with_name("test")
    extensions = {".png", ".jpg", ".jpeg"}
    training = []
    for label, name in enumerate(classes):
        images = sorted(p for p in (train / name).iterdir() if p.is_file() and p.suffix.lower() in extensions)
        if len(images) != 100:
            raise ValueError(f"{name}: expected 100 labeled images")
        training.extend((str(p), str(p), label) for p in images)
    images = sorted(p for p in test.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if len(images) != 378:
        raise ValueError("Expected 378 unlabeled test images")
    testing = [(str(p), str(p), None) for p in images]
    templates = list(root.rglob("submission-for-task2.csv"))
    if len(templates) != 1:
        raise ValueError("Expected one official submission-for-task2.csv")
    with templates[0].open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "Category"]:
            raise ValueError("Template columns must be ID,Category")
        ids = [row["ID"] for row in reader]
    if len(ids) != len(images) or set(ids) != {p.name for p in images}:
        raise ValueError("Template IDs must cover all 378 test images once")
    return train, test, training, testing, ids


def exg_plant_box(pil_img, work=96, min_frac=0.004):
    """
    用 ExG(2G-R-B)超绿指数 + Otsu + 最大连通域,定位植物主体 bbox。
    返回 (l, t, r, b) 或 None。全程在低分辨率上做,速度很快。
    """
    w, h = pil_img.size
    small = pil_img.convert("RGB").resize((work, work), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    exg = 2 * g - r - b

    # Otsu 阈值
    hist, edges = np.histogram(exg, bins=64)
    idx = np.arange(len(hist))
    best_t, best_v = float(edges[0]), -1.0
    for i in range(1, len(hist) - 1):
        w0, w1 = hist[:i].sum(), hist[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:i] * idx[:i]).sum() / w0
        m1 = (hist[i:] * idx[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best_v:
            best_v, best_t = v, edges[i]

    mask = exg > max(best_t, 5.0)
    if mask.sum() < work * work * min_frac:
        return None

    # 最大连通域(BFS,网格很小,足够快)
    from collections import deque
    seen = np.zeros_like(mask, dtype=bool)
    best = (0, 0, work - 1, work - 1, 0)
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        n = 0
        l = r_ = x0
        t = b_ = y0
        while q:
            y, x = q.popleft()
            n += 1
            l, r_ = min(l, x), max(r_, x)
            t, b_ = min(t, y), max(b_, y)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < work and 0 <= nx < work and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if n > best[4]:
            best = (l, t, r_, b_, n)

    l, t, r_, b_, n = best
    if n < work * work * min_frac:
        return None
    sc = w / work
    return (l * sc, t * sc, (r_ + 1) * sc, (b_ + 1) * sc)


def otsu_threshold(vals, nbins=64):
    """一维 Otsu 阈值(numpy 实现)"""
    hist, edges = np.histogram(vals, bins=nbins)
    idx = np.arange(len(hist))
    best_t, best_v = float(edges[0]), -1.0
    for i in range(1, len(hist) - 1):
        w0, w1 = hist[:i].sum(), hist[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:i] * idx[:i]).sum() / w0
        m1 = (hist[i:] * idx[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best_v:
            best_v, best_t = v, edges[i]
    return best_t


def remove_background(pil_img, bg=0.35, min_frac=0.004, dilate=2):
    """
    【实测最有效的数据侧处理】用 ExG 超绿指数把非植物像素压暗。
    实测(resnet18 冻结特征 + 逻辑回归,5 折 CV):
        原图 macro-F1 = 0.601  ->  去背景 macro-F1 = 0.7675
    原因:基质颗粒纹理是强干扰,压暗后模型注意力集中到叶片形态上。
    """
    a = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    exg = 2 * g - r - b
    t = otsu_threshold(exg.ravel())
    mask = exg > max(t, 6.0)
    if mask.mean() < min_frac:          # 没检到植物,原样返回(不破坏样本)
        return pil_img
    m = mask.copy()
    for _ in range(max(0, dilate)):     # 轻度膨胀,避免切掉叶片边缘
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    out = a.copy()
    out[~m] = out[~m] * bg + 100.0 * (1.0 - bg)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def plant_centered_crop(pil_img, plant_ratio=0.55, min_frac=0.30, max_frac=1.0):
    """
    【数据侧核心】按植物尺度归一化裁剪:
    先找到植物 bbox,再以它为中心取一块正方形,使植物占该正方形的 plant_ratio。
    这样不同原图尺寸的样本,裁完以后"植物大小"就统一了。
    检测失败则退回原图中心裁剪(保守)。
    """
    w, h = pil_img.size
    box = exg_plant_box(pil_img)
    if box is None:
        side = min(w, h)
        cx, cy = w / 2, h / 2
    else:
        l, t, r_, b_ = box
        cx, cy = (l + r_) / 2, (t + b_) / 2
        side = max(r_ - l, b_ - t) / max(plant_ratio, 1e-3)

    side = float(np.clip(side, min(w, h) * min_frac, min(w, h) * max_frac))

    # 取正方形并 clamp 到图像内
    l2 = int(round(max(0, min(cx - side / 2, w - side))))
    t2 = int(round(max(0, min(cy - side / 2, h - side))))
    r2 = int(round(min(w, l2 + side)))
    b2 = int(round(min(h, t2 + side)))
    if r2 - l2 < 8 or b2 - t2 < 8:
        return pil_img
    return pil_img.crop((l2, t2, r2, b2))


# ============================================================
# 2. 数据准备(带磁盘缓存)
# ============================================================

def build_cache(train_dir, test_dir, cfg, cache_dir):
    """
    预处理一次,把结果写成 jpg 缓存:
      - 可选的植物尺度归一化裁剪
      - 最长边限制到 cfg['cache_size'](小图不放大,避免无意义插值)
    返回 (train_items, test_items),每项是 (缓存文件路径, 标签或 None)
    """
    os.makedirs(cache_dir, exist_ok=True)
    classes = sorted([d for d in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, d))])
    train_items, test_items = [], []

    for ci, cname in enumerate(classes):
        files = sorted(Path(train_dir, cname).glob("*"))
        files = [f for f in files if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        for f in files:
            out = os.path.join(cache_dir, f"tr_{ci}_{f.stem}.jpg")
            train_items.append((str(f), out, ci))

    test_files = sorted([f for f in Path(test_dir).glob("*")
                         if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    for f in test_files:
        out = os.path.join(cache_dir, f"te_{f.name}")
        out = os.path.splitext(out)[0] + ".jpg"
        test_items.append((str(f), out, None))

    def _process(src, dst, idx, total, tag):
        if os.path.exists(dst):
            return
        im = Image.open(src).convert("RGB")
        # 先缩放(小图不放大),再做前景处理 —— 省时间
        w, h = im.size
        if max(w, h) > cfg["cache_size"]:
            sc = cfg["cache_size"] / max(w, h)
            # 用 LANCZOS 下采样:细长叶片用 BILINEAR 容易丢细节/出锯齿
            im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)
        if cfg["use_plant_crop"]:
            im = plant_centered_crop(im, cfg["plant_ratio"],
                                     cfg["crop_min_frac"], cfg["crop_max_frac"])
        if cfg["use_bg_removal"]:
            im = remove_background(im, bg=cfg["bg_darken"])
        im.save(dst, quality=95)
        if idx % 50 == 0:
            print(f"  [{tag}] {idx}/{total}", flush=True)

    total = len(train_items) + len(test_items)
    print(f"[缓存] 预处理 {total} 张图 -> {cache_dir}", flush=True)
    t0 = time.time()
    for i, (src, dst, _) in enumerate(train_items):
        _process(src, dst, i, total, "train")
    for j, (src, dst, _) in enumerate(test_items):
        _process(src, dst, len(train_items) + j, total, "test")
    print(f"[缓存] 完成,用时 {time.time()-t0:.1f}s", flush=True)
    return classes, train_items, test_items


# ============================================================
# 3. Dataset / 增强
# ============================================================

from functools import lru_cache


@lru_cache(maxsize=128)
def load_rgb_cached(path):
    # Exact decoded original pixels; do not resize or save lossy JPEGs.
    with Image.open(path) as source:
        return source.convert("RGB")


class PlantDS(Dataset):
    def __init__(self, items, img_size, train=True, aug_level="strong"):
        self.items = items
        self.img_size = img_size
        self.train = train
        self.aug = aug_level
        self.ra = None
        self._eval_cache = {}
        if train and aug_level == "strong":
            try:
                from torchvision.transforms import RandAugment
                self.ra = RandAugment(num_ops=2, magnitude=9)
            except Exception:
                self.ra = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if not self.train and i in self._eval_cache:
            return self._eval_cache[i]
        path, _, label = self.items[i]
        im = load_rgb_cached(path).copy()
        if self.train:
            w, h = im.size
            if self.aug == "none":
                # 无增强:整图直接缩放到目标尺寸(用于"模型能否拟合训练集"的诊断)
                im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
            elif self.aug == "light":
                # 轻增强:中心裁剪 + 翻转,不做随机尺度裁剪/颜色抖动
                side = min(w, h)
                x0, y0 = (w - side) // 2, (h - side) // 2
                im = im.crop((x0, y0, x0 + side, y0 + side))
                im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
                if random.random() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                # 强增强(默认):随机尺度裁剪 + RandAugment + 翻转旋转 + 颜色抖动
                s = random.uniform(0.35, 1.0)
                side = int(round(min(w, h) * math.sqrt(s)))
                side = max(side, 16)
                x0 = random.randint(0, max(0, w - side))
                y0 = random.randint(0, max(0, h - side))
                im = im.crop((x0, y0, x0 + side, y0 + side))
                im = im.resize((self.img_size, self.img_size), Image.BILINEAR)

                if self.ra is not None and random.random() < 0.5:
                    im = self.ra(im)

                if random.random() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() < 0.5:
                    im = im.transpose(Image.FLIP_TOP_BOTTOM)
                k = random.randint(0, 3)
                if k:
                    im = im.rotate(90 * k, expand=True)          # 俯拍图,旋转不变
                if random.random() < 0.3:
                    # 颜色抖动(不同光照条件)
                    im = ImageEnhance.Brightness(im).enhance(random.uniform(0.8, 1.2))
                    im = ImageEnhance.Contrast(im).enhance(random.uniform(0.8, 1.2))
                    im = ImageEnhance.Color(im).enhance(random.uniform(0.8, 1.2))
                if random.random() < 0.2:
                    gain = random.uniform(0.95, 1.05)
                    g = np.asarray(im, dtype=np.float32) * gain
                    im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
        else:
            # 验证/测试:整图(短边)缩放到 img_size,再中心裁剪
            w, h = im.size
            sc = self.img_size / min(w, h)
            nw, nh = max(self.img_size, int(round(w * sc))), max(self.img_size, int(round(h * sc)))
            im = im.resize((nw, nh), Image.BILINEAR)
            left = (nw - self.img_size) // 2
            top = (nh - self.img_size) // 2
            im = im.crop((left, top, left + self.img_size, top + self.img_size))

        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
            np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        y = -1 if label is None else label
        sample = (x, y, self.items[i][0])
        if not self.train:
            self._eval_cache[i] = sample
        return sample


# ============================================================
# 4. 模型 / 损失 / 优化器
# ============================================================

def _hf_reachable(timeout=6):
    """快速探测 HuggingFace(或 HF_ENDPOINT 镜像)是否可达,避免 timm 卡在重试上"""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    host = endpoint.split("//")[-1].split("/")[0]
    import socket
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except Exception:
        return False


def _torchvision_fallback(n_classes):
    """HF 不可达时的兜底:用 torchvision 的预训练 resnet18
    (权重仅 46MB,download.pytorch.org 通常可达;保底能跑,不追求最优)"""
    import torchvision
    m = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    print("[回退] 使用 torchvision resnet18 (IMAGENET1K_V1)", flush=True)
    return m


def build_model(cfg, n_classes):
    """
    优先 timm(权重来自 HuggingFace);若 HF 不可达或加载失败,自动回退 torchvision,
    保证脚本在任何网络环境下都能跑起来。
    """
    import timm
    pretrained = os.environ.get("PRETRAINED", "1") != "0"   # 离线/冒烟测试时可设 PRETRAINED=0
    if pretrained and not _hf_reachable():
        print("[提示] HuggingFace 不可达,跳过 timm,直接回退 torchvision 预训练权重", flush=True)
        return _torchvision_fallback(n_classes)
    try:
        return timm.create_model(
            cfg["arch"], pretrained=pretrained, num_classes=n_classes,
            drop_rate=cfg["drop_rate"], drop_path_rate=cfg["drop_path_rate"],
        )
    except Exception as e:
        print(f"[提示] timm 加载 {cfg['arch']} 失败({type(e).__name__}),回退 torchvision", flush=True)
        return _torchvision_fallback(n_classes)


def param_groups(model, base_lr, wd):
    """backbone 用较小 lr,分类头用较大 lr"""
    head_names = ("head", "fc", "classifier")
    head, body = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if any(n.startswith(h) for h in head_names) else body).append(p)
    return [
        {"params": body, "lr": base_lr * 0.3, "weight_decay": wd},
        {"params": head, "lr": base_lr, "weight_decay": wd},
    ]


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup = {}

    def update(self, model, step=None):
        d = self.decay
        if step is not None:
            # warmup:早期让 EMA 快速跟上模型权重。
            # 否则在训练步数不多时(小数据集很常见),EMA 仍停留在初始权重上,
            # 而分类头是随机初始化的 —— 评估就等价于随机猜(本任务踩过这个坑)。
            d = min(self.decay, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def apply_to(self, model):
        self.backup = {}
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.backup[k] = v.detach().clone()
                v.copy_(self.shadow[k].to(v.dtype))

    def restore(self, model):
        for k, v in model.state_dict().items():
            if k in self.backup:
                v.copy_(self.backup[k])
        self.backup = {}


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def mixup_cutmix_loss(crit, model, x, y, cfg):
    """对一个 batch 施加 mixup 或 cutmix 后前向。返回 (loss, y_a, y_b, lam)"""
    if random.random() > cfg["mix_prob"]:
        return crit(model(x), y), y, y, 1.0
    index = torch.randperm(x.size(0), device=x.device)
    if cfg["cutmix_alpha"] > 0 and random.random() < 0.5:
        lam = float(np.random.beta(cfg["cutmix_alpha"], cfg["cutmix_alpha"]))
        x1, y1, x2, y2 = rand_bbox(x.size(), lam)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[index][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
    else:
        lam = float(np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"])) \
            if cfg["mixup_alpha"] > 0 else 1.0
        x = lam * x + (1 - lam) * x[index]
    y_b = y[index]      # 另一路的真实标签(注意:是 y[index],不是置换索引 index 本身)
    return lam * crit(model(x), y) + (1 - lam) * crit(model(x), y_b), y, y_b, lam


def macro_f1(y_true, y_pred, n_classes):
    """不依赖 sklearn 的 macro-F1 实现"""
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


# ============================================================
# 5. 训练 / 验证
# ============================================================

@torch.no_grad()
def predict(model, loader, device, tta=False):
    """返回 (概率矩阵, 标签数组)。tta=True 时做 4 种方向的增强平均。"""
    model.eval()
    nc = getattr(model, "num_classes", None)
    if nc is None:
        for attr in ("head", "fc", "classifier"):
            m = getattr(model, attr, None)
            if m is not None and hasattr(m, "out_features"):
                nc = m.out_features
                break
    nc = int(nc or 5)
    probs, ys = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            if not tta:
                p = F.softmax(model(x), dim=1)
            else:
                p = torch.zeros(x.size(0), nc, device=x.device)
                for t in range(4):
                    xr = torch.rot90(x, k=t, dims=(2, 3))
                    p += F.softmax(model(xr), dim=1)
                    p += F.softmax(model(xr.flip(3)), dim=1)
                p /= 8.0
        probs.append(p.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def run_fold(fold, tr_items, va_items, cfg, classes, device, out_dir):
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    tr_loader = DataLoader(PlantDS(tr_items, cfg["img_size"], True, cfg.get("aug_level", "strong")),
                           batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=cfg["num_workers"], drop_last=True,
                           persistent_workers=cfg["num_workers"] > 0)
    va_loader = DataLoader(PlantDS(va_items, cfg["img_size"], False),
                           batch_size=max(8, cfg["batch_size"]), shuffle=False,
                           num_workers=cfg["num_workers"],
                           persistent_workers=cfg["num_workers"] > 0)

    model = build_model(cfg, len(classes)).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    opt = AdamW(param_groups(model, cfg["lr"], cfg["weight_decay"]))
    steps_per_epoch = max(1, len(tr_loader))
    total_steps = steps_per_epoch * cfg["epochs"]
    warm = steps_per_epoch * cfg["warmup_epochs"]

    def lr_lambda(step):
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * prog)) * 0.99 + 0.01

    sched = LambdaLR(opt, lr_lambda)
    ema = EMA(model, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_f1, best_state, step = -1.0, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        t0, losses, seen = time.time(), 0.0, 0
        for x, y, _ in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                loss, ya, yb, lam = mixup_cutmix_loss(crit, model, x, y, cfg)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if ema:
                ema.update(model, step)
            losses += loss.item() * x.size(0)
            seen += x.size(0)

        # 评估:分别用"EMA 权重"和"当前权重",取 val F1 更高的那套
        # (EMA 在训练早期还没 warmup 好,用它评估会低得离谱 —— 这个坑踩过)
        val_p, val_y = predict(model, va_loader, device, tta=False)
        val_f1, per = macro_f1(val_y, val_p.argmax(1), len(classes))
        val_acc = float((val_p.argmax(1) == val_y).mean())
        best_src = "raw"
        if ema:
            ema.apply_to(model)
            p_ema, _ = predict(model, va_loader, device, tta=False)
            ema.restore(model)
            f1_ema, _ = macro_f1(val_y, p_ema.argmax(1), len(classes))
            if f1_ema > val_f1:
                val_f1 = f1_ema
                val_acc = float((p_ema.argmax(1) == val_y).mean())
                best_src = "ema"

        mark = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            if best_src == "ema":
                best_state = {}
                for k, v in model.state_dict().items():
                    best_state[k] = (ema.shadow[k] if k in ema.shadow
                                     else v).detach().cpu().clone()
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = f"  <- best({best_src})"
        print(f"  ep {ep+1:02d}/{cfg['epochs']} | loss {losses/max(1,seen):.4f} | "
              f"val acc {val_acc:.4f} F1 {val_f1:.4f} | {time.time()-t0:.0f}s{mark}", flush=True)

    print(f"  [fold {fold}] best val macro-F1 = {best_f1:.4f}", flush=True)
    return model, best_state, best_f1


# ============================================================
# 6. 主流程
# ============================================================

def main(quick=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟测试:1 折 / 2 epoch / 160px")
    args, _ = ap.parse_known_args()
    quick = quick or args.quick

    cfg = dict(CFG)
    if quick:
        cfg.update(img_size=int(os.environ.get("Q_SIZE", 160)),
                   cache_size=int(os.environ.get("Q_CACHE", 200)),
                   epochs=int(os.environ.get("Q_EPOCHS", 2)),
                   n_folds=int(os.environ.get("Q_FOLDS", 2)),
                   batch_size=int(os.environ.get("Q_BS", 8)),
                   num_workers=0,
                   arch=os.environ.get("Q_ARCH", "resnet18.a1_in1k"),
                   max_folds=int(os.environ.get("Q_MAXF", 0)) or None,
                   tta=os.environ.get("Q_TTA", "0") == "1")
        cfg["out_dir"] = os.environ.get("Q_OUT", "./plant_out_quick")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"设备: {device} | 分辨率: {cfg['img_size']} | 模型: {cfg['arch']}", flush=True)
    if device.type == "cpu":
        print("⚠ 没有 GPU,训练会非常慢(仅供验证代码正确性)", flush=True)

    train_dir, test_dir = find_dirs()
    print(f"train: {train_dir}\ntest : {test_dir}", flush=True)

    cache_dir = os.path.join(cfg["out_dir"], "img_cache")
    classes, train_items, test_items = build_cache(train_dir, test_dir, cfg, cache_dir)
    print(f"类别({len(classes)}): {classes}", flush=True)
    print(f"训练样本 {len(train_items)} | 测试样本 {len(test_items)}", flush=True)

    labels = np.array([it[2] for it in train_items])

    # ---- 交叉验证 ----
    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
        splits = list(skf.split(np.zeros(len(labels)), labels))
    except ImportError:      # 没装 sklearn 就自己分层切
        splits = []
        rng = np.random.RandomState(cfg["seed"])
        for c in range(len(classes)):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            splits.append(idx)
        folds = [[] for _ in range(cfg["n_folds"])]
        for c, idx in enumerate(splits):
            for j, k in enumerate(idx):
                folds[j % cfg["n_folds"]].append(k)
        splits = []
        for f in range(cfg["n_folds"]):
            va = np.array(folds[f])
            tr = np.setdiff1d(np.arange(len(labels)), va)
            splits.append((tr, va))

    oof = np.zeros((len(train_items), len(classes)), dtype=np.float32)
    oof_covered = np.zeros(len(train_items), dtype=bool)
    ckpts, fold_f1s, fold_train_accs, fold_train_f1s = [], [], [], []
    max_folds = int(cfg.get("max_folds") or 0) or len(splits)
    for fold, (tr_idx, va_idx) in enumerate(splits[:max_folds]):
        print(f"\n{'='*60}\nFold {fold+1}/{len(splits)}  train={len(tr_idx)} val={len(va_idx)}\n{'='*60}", flush=True)
        tr_items = [train_items[i] for i in tr_idx]
        va_items = [train_items[i] for i in va_idx]
        model, best_state, best_f1 = run_fold(fold, tr_items, va_items, cfg, classes, device, cfg["out_dir"])
        fold_f1s.append(best_f1)

        # 载入最佳权重,对验证集做 TTA 预测 -> OOF
        if best_state is not None:
            model.load_state_dict(best_state)

        # 训练集指标(便于直接核查"训练集得分",不开 TTA)
        tr_eval = DataLoader(PlantDS(tr_items, cfg["img_size"], False),
                             batch_size=max(8, cfg["batch_size"]), shuffle=False,
                             num_workers=cfg["num_workers"])
        trp, tr_y = predict(model, tr_eval, device, tta=False)
        tr_acc = float((trp.argmax(1) == tr_y).mean())
        tr_f1, _ = macro_f1(tr_y, trp.argmax(1), len(classes))
        fold_train_accs.append(tr_acc)
        fold_train_f1s.append(tr_f1)
        print(f"  [fold {fold}] 训练集: acc = {tr_acc:.4f}   macro-F1 = {tr_f1:.4f}", flush=True)

        va_loader = DataLoader(PlantDS(va_items, cfg["img_size"], False),
                               batch_size=max(8, cfg["batch_size"]), shuffle=False,
                               num_workers=cfg["num_workers"])
        vp, vy = predict(model, va_loader, device, tta=cfg["tta"])
        oof[va_idx] = vp
        oof_covered[va_idx] = True
        # 测试集预测(每折保存,最后平均)
        te_loader = DataLoader(PlantDS(test_items, cfg["img_size"], False),
                               batch_size=max(8, cfg["batch_size"]), shuffle=False,
                               num_workers=cfg["num_workers"])
        tp, _ = predict(model, te_loader, device, tta=cfg["tta"])
        np.save(os.path.join(cfg["out_dir"], f"test_prob_fold{fold}.npy"), tp)
        ck = os.path.join(cfg["out_dir"], f"best_fold{fold}.pth")
        torch.save({"state": best_state, "classes": classes, "arch": cfg["arch"],
                    "img_size": cfg["img_size"]}, ck)
        ckpts.append(ck)
        del model, te_loader, va_loader

    # ---- OOF 报告 ----
    print(f"\n{'='*60}\n交叉验证结果\n{'='*60}")
    print(f"各折 macro-F1: {[round(x,4) for x in fold_f1s]}")
    print(f"平均 macro-F1  = {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    if fold_train_accs:
        print(f"训练集(各折平均): accuracy = {np.mean(fold_train_accs):.4f}   "
              f"macro-F1 = {np.mean(fold_train_f1s):.4f}")

    # 只统计真正跑过的折(部分折实验时,未覆盖样本概率是全零,会把指标算成垃圾数据)
    cov = np.where(oof_covered)[0]
    if len(cov) < len(labels):
        print(f"\n[注意] 本次只跑了 {max_folds}/{len(splits)} 折,"
              f"OOF 指标在 {len(cov)}/{len(labels)} 个样本上计算")
    oof_pred = oof[cov].argmax(1)
    oof_f1, per_f1 = macro_f1(labels[cov], oof_pred, len(classes))
    oof_acc = float((oof_pred == labels[cov]).mean())
    print(f"\n[OOF] accuracy = {oof_acc:.4f}   macro-F1 = {oof_f1:.4f}")
    print("各类 F1:")
    for c, v in zip(classes, per_f1):
        print(f"  {c:<20} {v:.4f}")
    np.save(os.path.join(cfg["out_dir"], "oof_prob.npy"), oof)
    with open(os.path.join(cfg["out_dir"], "cv_report.json"), "w") as f:
        json.dump({"classes": classes, "fold_f1": fold_f1s,
                   "oof_macro_f1": oof_f1, "oof_accuracy": oof_acc,
                   "per_class_f1": dict(zip(classes, per_f1)),
                   "cfg": {k: v for k, v in cfg.items()}}, f, indent=2, ensure_ascii=False)

    # ---- 测试集集成 + 提交 ----
    test_probs = []
    for fold in range(len(splits)):
        p = os.path.join(cfg["out_dir"], f"test_prob_fold{fold}.npy")
        if os.path.exists(p):
            test_probs.append(np.load(p))
    if test_probs:
        mean_p = np.mean(test_probs, axis=0)
        pred_names = [test_items[i][0] for i in range(len(test_items))]
        ids = [os.path.basename(p) for p in pred_names]
        sub_path = os.path.join(cfg["out_dir"], "submission.csv")
        with open(sub_path, "w", encoding="utf-8", newline="") as f:
            f.write("ID,Category\r\n")
            for i in np.argsort(ids):
                f.write(f"{ids[i]},{classes[int(mean_p[i].argmax())]}\r\n")
        print(f"\n提交文件已写入: {sub_path}")
        print("预测分布:", dict(Counter(classes[int(x)] for x in mean_p.argmax(1))))

    print(f"\n所有产物在: {cfg['out_dir']}")


if __name__ == "__main__":
    main()
