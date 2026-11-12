"""Small-data U-Net, image-level validation and random GPU patch training."""
from __future__ import annotations

import contextlib
import csv
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F

from retinal import image_paths, index_images, write_json


def setup(seed=42, device="auto"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        # Fixed-size batches permit autotuning; exact cross-device reproducibility
        # is not promised. Seeds, versions and all IDs are saved with each run.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda_build": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seeded": True, "bitwise_determinism": False}


def preprocess(path, size=512):
    rgb = np.asarray(Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    green = rgb[..., 1]
    contrast = np.clip((green - ndimage.gaussian_filter(green, sigma=8.0)) * 4.0 + 0.5, 0, 1)
    features = np.concatenate((rgb, contrast[..., None]), axis=-1).transpose(2, 0, 1)
    fov = ndimage.binary_fill_holes(ndimage.gaussian_filter(green, 1.0) > 0.02)
    fov = ndimage.binary_dilation(fov, iterations=2)
    return np.ascontiguousarray(features), fov


def read_label(path, size=512):
    label = np.asarray(Image.open(path).convert("L").resize(
        (size, size), Image.Resampling.NEAREST))
    return (label >= 128).astype(np.float32)


def get_split(root, fold, seed):
    ids = [p.stem for p in image_paths(Path(root) / "train/image")]
    if fold == "all":
        return ids, []
    fold = int(fold)
    if fold not in range(5):
        raise ValueError("fold must be 0,1,2,3,4 or all")
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    val = shuffled[fold::5]
    train = [key for key in ids if key not in val]
    if set(train) & set(val) or set(train) | set(val) != set(ids):
        raise AssertionError("Invalid image-level split")
    return train, val


def load_training(root, ids, size=512):
    image_index = index_images(Path(root) / "train/image")
    label_index = index_images(Path(root) / "train/label")
    images, labels, fovs = [], [], []
    for key in ids:
        x, fov = preprocess(image_index[key], size)
        images.append(x)
        labels.append(read_label(label_index[key], size)[None])
        fovs.append(fov)
    return np.stack(images), np.stack(labels), np.stack(fovs)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.width = width
        channels = [width * 2 ** i for i in range(4)]
        self.encoders = nn.ModuleList([ConvBlock(4, channels[0])] +
            [ConvBlock(channels[i - 1], channels[i]) for i in range(1, 4)])
        self.decoders = nn.ModuleList([
            ConvBlock(channels[i] + channels[i - 1], channels[i - 1]) for i in range(3, 0, -1)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x)


class PatchSampler:
    """All training images stay on the chosen device; no worker/split ambiguity."""
    def __init__(self, images, labels, device, patch=160, batch=8):
        self.images = torch.from_numpy(images).to(device)
        self.labels = torch.from_numpy(labels).to(device)
        self.device, self.patch, self.batch = device, patch, batch
        self.size = images.shape[-1]
        if not 32 <= patch <= self.size or patch % 8:
            raise ValueError("patch must be divisible by 8 and between 32 and image size")

    def sample(self):
        n = self.batch
        indices = torch.randint(len(self.images), (n,), device=self.device)
        angle = torch.rand(n, device=self.device) * (2 * math.pi)
        scale = (torch.rand(n, device=self.device) * 0.4 + 0.8) * (self.patch / self.size)
        cosine, sine = angle.cos() * scale, angle.sin() * scale
        mirror = torch.randint(2, (n,), device=self.device).float() * 2 - 1
        extent = (cosine.abs() + sine.abs()).clamp(max=1)
        centers = (torch.rand(n, 2, device=self.device) * 2 - 1) * (1 - extent[:, None])
        theta = torch.zeros(n, 2, 3, device=self.device)
        theta[:, 0, 0], theta[:, 0, 1] = cosine * mirror, -sine
        theta[:, 1, 0], theta[:, 1, 1] = sine * mirror, cosine
        theta[:, :, 2] = centers
        grid = F.affine_grid(theta, (n, 4, self.patch, self.patch), align_corners=False)
        x = F.grid_sample(self.images[indices], grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        y = F.grid_sample(self.labels[indices], grid, mode="nearest", padding_mode="zeros", align_corners=False)
        # Jitter only RGB; the contrast channel represents local differences.
        brightness = torch.rand(n, 1, 1, 1, device=self.device) * 0.3 + 0.85
        contrast = torch.rand(n, 1, 1, 1, device=self.device) * 0.3 + 0.85
        offsets = (torch.rand(n, 3, 1, 1, device=self.device) - 0.5) * 0.04
        x[:, :3] = ((x[:, :3] - 0.3) * contrast + 0.3) * brightness + offsets
        x = (x.clamp(0, 1) - 0.5) / 0.25
        return x.contiguous(memory_format=torch.channels_last), y


def segmentation_loss(logits, target):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (probability * target).sum(dims) + 1) / (
        probability.sum(dims) + target.sum(dims) + 1)
    return 0.5 * bce + 0.5 * (1 - dice.mean())


def amp_context(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()


@torch.inference_mode()
def predict_one(model, image, device, tta=False):
    model.eval()
    x = torch.from_numpy(image[None]).to(device)
    x = ((x - 0.5) / 0.25).contiguous(memory_format=torch.channels_last)
    # Four flips are calibrated on validation by the same inference path.
    transforms = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    probabilities = []
    for dims in transforms:
        inp = torch.flip(x, dims) if dims else x
        with amp_context(device):
            logits = model(inp)
        probability = logits.float().sigmoid()
        if dims:
            probability = torch.flip(probability, dims)
        probabilities.append(probability)
    return torch.stack(probabilities).mean(0)[0, 0].cpu().numpy()


def dice_per_image(probabilities, targets, threshold):
    prediction = np.asarray(probabilities) >= threshold
    targets = np.asarray(targets, dtype=bool)
    axes = tuple(range(1, prediction.ndim))
    intersection = (prediction & targets).sum(axis=axes, dtype=np.float64)
    denominator = prediction.sum(axis=axes) + targets.sum(axis=axes)
    return np.divide(2 * intersection, denominator, out=np.ones_like(intersection), where=denominator > 0)


def choose_threshold(probabilities, targets):
    results = []
    for threshold in np.arange(0.20, 0.801, 0.025):
        scores = dice_per_image(probabilities, targets, float(threshold))
        results.append({"threshold": round(float(threshold), 3), "mean_dice": float(scores.mean())})
    # Tie-break toward 0.5 for a stable default.
    best = max(results, key=lambda row: (row["mean_dice"], -abs(row["threshold"] - 0.5)))
    return best["threshold"], results


def save_checkpoint(path, model, optimizer, scheduler, epoch, config, split, score):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "config": config,
                "split": split, "val_dice_at_0_5": score}, path)


def load_model(path, device):
    # These are checkpoints produced by this project, not external pickles.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = UNet(int(checkpoint["config"]["width"])).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def check(args):
    device = setup(args.seed, args.device)
    train_ids, val_ids = get_split(args.data, "0", args.seed)
    x, y, _ = load_training(args.data, train_ids[:2])
    sampler = PatchSampler(x, y, device, args.patch, args.batch)
    model = UNet(args.width).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    initial = model.head.weight.detach().clone()
    losses, elapsed = [], []
    for _ in range(8):
        t0 = time.perf_counter()
        inputs, targets = sampler.sample()
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            logits = model(inputs)
            loss = segmentation_loss(logits, targets)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise AssertionError("Nonfinite gradient")
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - t0)
    if torch.equal(initial, model.head.weight.detach()):
        raise AssertionError("Optimizer did not update model parameters")
    prediction = predict_one(model, x[0], device)
    if prediction.shape != (512, 512) or not np.isfinite(prediction).all():
        raise AssertionError("Invalid prediction")
    report = {"environment": environment(device), "train_ids": train_ids, "val_ids": val_ids,
              "parameters": sum(p.numel() for p in model.parameters()), "losses": losses,
              "seconds_per_step_after_warmup": float(np.mean(elapsed[3:])),
              "prediction_shape": list(prediction.shape), "checks": "passed",
              "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None}
    write_json(Path(args.out) / "model_check.json", report)
    print(json.dumps(report, indent=2), flush=True)


def train(args):
    if args.epochs < 1 or args.steps < 1 or args.validate_every < 1 or args.batch < 1 or args.width < 1:
        raise ValueError("epochs, steps, validate-every, batch and width must be positive")
    device = setup(args.seed, args.device)
    out = Path(args.out) / f"fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists() or (out / "last.pt").exists():
        raise FileExistsError(f"Existing training run: {out}; choose another --out to preserve it")
    train_ids, val_ids = get_split(args.data, args.fold, args.seed)
    images, labels, _ = load_training(args.data, train_ids)
    sampler = PatchSampler(images, labels, device, args.patch, args.batch)
    if val_ids:
        val_x, val_y, val_fov = load_training(args.data, val_ids)
        val_y = val_y[:, 0]
    model = UNet(args.width).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}
    config["environment"] = environment(device)
    split = {"train_ids": train_ids, "val_ids": val_ids, "fold": args.fold, "seed": args.seed}
    write_json(out / "config.json", config)
    write_json(out / "split.json", split)
    print(f"TRAIN device={device} fold={args.fold} train={train_ids} val={val_ids}", flush=True)
    best, best_epoch, stale = -1.0, 0, 0
    start = time.perf_counter()
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "lr", "val_dice_0_5", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            losses = []
            for _ in range(args.steps):
                x, y = sampler.sample()
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device):
                    loss = segmentation_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}")
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            score = None
            if val_ids and (epoch % args.validate_every == 0 or epoch == args.epochs or epoch == 1):
                p = np.stack([predict_one(model, x, device) * fov for x, fov in zip(val_x, val_fov)])
                score = float(dice_per_image(p, val_y, 0.5).mean())
                if score > best:
                    best, best_epoch, stale = score, epoch, 0
                    save_checkpoint(out / "best.pt", model, optimizer, scheduler, epoch, config, split, score)
                else:
                    stale += 1
            if not val_ids and epoch == args.epochs:
                best_epoch = epoch
                save_checkpoint(out / "best.pt", model, optimizer, scheduler, epoch, config, split, None)
            seconds = time.perf_counter() - epoch_start
            writer.writerow({"epoch": epoch, "loss": np.mean(losses), "lr": lr,
                             "val_dice_0_5": score, "seconds": seconds})
            f.flush()
            score_text = f" val_dice={score:.5f} best={best:.5f}@{best_epoch}" if score is not None else ""
            print(f"epoch {epoch:03d}/{args.epochs} loss={np.mean(losses):.5f}{score_text} time={seconds:.1f}s", flush=True)
            write_json(out / "progress.json", {"epoch": epoch, "total_epochs": args.epochs,
                       "best_epoch": best_epoch, "best_val_dice_0_5": best if val_ids else None,
                       "elapsed_seconds": time.perf_counter() - start})
            if val_ids and args.patience > 0 and stale >= args.patience:
                print("Early stopping: validation-check patience reached", flush=True)
                break
        save_checkpoint(out / "last.pt", model, optimizer, scheduler, epoch, config, split, score)
    write_json(out / "training_summary.json", {"fold": args.fold, "epochs_completed": epoch,
               "best_epoch": best_epoch, "best_val_dice_0_5": best if val_ids else None,
               "elapsed_seconds": time.perf_counter() - start,
               "validation_type": "checkpoint-selection validation, not independent test" if val_ids else "none: all-data refit"})
    if val_ids:
        evaluate_checkpoint(out / "best.pt", args.data, device, out, tta=True)


def evaluate_checkpoint(checkpoint_path, root, device, out, tta=True):
    out = Path(out)
    model, checkpoint = load_model(checkpoint_path, device)
    ids = checkpoint["split"]["val_ids"]
    if not ids:
        raise ValueError("All-data model has no held-out validation split")
    images, targets, fovs = load_training(root, ids)
    probabilities = np.stack([predict_one(model, image, device, tta) * fov
                              for image, fov in zip(images, fovs)])
    targets = targets[:, 0].astype(bool)
    threshold, sweep = choose_threshold(probabilities, targets)
    per_image = dice_per_image(probabilities, targets, threshold)
    fixed_scores = dice_per_image(probabilities, targets, 0.5)
    rows = [{"id": key, "vessel_dice_tuned": float(score), "vessel_dice_0_5": float(fixed)}
            for key, score, fixed in zip(ids, per_image, fixed_scores)]
    np.savez_compressed(out / "validation_predictions.npz", ids=np.array(ids),
                        probabilities=probabilities.astype(np.float32), targets=targets)
    report = {"checkpoint": str(checkpoint_path), "epoch": checkpoint["epoch"], "val_ids": ids,
              "train_ids": checkpoint["split"]["train_ids"], "tta": tta, "threshold": threshold,
              "mean_vessel_dice": float(per_image.mean()), "mean_vessel_dice_0_5": float(fixed_scores.mean()),
              "per_image": rows, "threshold_sweep": sweep,
              "metric": "mean per-image vessel Dice on full 512x512 nearest-resized masks",
              "selection_note": "Same held-out images select checkpoint and threshold; this is tuning validation, not independent test or Kaggle score."}
    write_json(out / "validation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "threshold_sweep"}, indent=2), flush=True)
    return report


def evaluate(args):
    device = setup(args.seed, args.device)
    evaluate_checkpoint(args.checkpoint, args.data, device, args.out, not args.no_tta)


def calibrate(args):
    """Combine only held-out predictions, never train-set ensemble predictions."""
    out = Path(args.out)
    wanted = [p.stem for p in image_paths(Path(args.data) / "train/image")]
    records, folds, data = {}, [], []
    for fold in args.folds:
        directory = out / f"fold_{fold}"
        report = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
        saved = np.load(directory / "validation_predictions.npz", allow_pickle=False)
        if saved["probabilities"].dtype != np.float32:
            raise ValueError(f"Fold {fold} has legacy float16 probabilities; rerun evaluate from its saved checkpoint")
        ids = saved["ids"].tolist()
        if ids != report["val_ids"] or set(ids) & set(report["train_ids"]):
            raise ValueError(f"Invalid held-out provenance for fold {fold}")
        expected_train, expected_val = get_split(args.data, str(fold), args.seed)
        if expected_train != report["train_ids"] or expected_val != ids:
            raise ValueError(f"Fold {fold} was generated with a different split/seed")
        if not report["tta"]:
            raise ValueError("Calibration requires the same four-flip TTA used for final predictions")
        for key, p, y in zip(ids, saved["probabilities"], saved["targets"]):
            if key in records:
                raise ValueError(f"Duplicate held-out prediction: {key}")
            records[key] = (p.astype(np.float32), y)
        folds.append({"fold": fold, "best_epoch": report["epoch"],
                      "mean_vessel_dice_0_5": report["mean_vessel_dice_0_5"], "val_ids": ids})
    if set(records) != set(wanted):
        raise ValueError("Complete 5-fold OOF calibration requires exactly one held-out prediction for all 20 images")
    probabilities = np.stack([records[key][0] for key in wanted])
    targets = np.stack([records[key][1] for key in wanted])
    threshold, sweep = choose_threshold(probabilities, targets)
    scores = dice_per_image(probabilities, targets, threshold)
    fixed = dice_per_image(probabilities, targets, 0.5)
    report = {"n_images": len(wanted), "threshold": threshold, "mean_vessel_dice": float(scores.mean()),
              "mean_vessel_dice_0_5": float(fixed.mean()), "std_vessel_dice": float(scores.std(ddof=1)),
              "per_image": [{"id": key, "vessel_dice_tuned": float(score), "vessel_dice_0_5": float(base)}
                            for key, score, base in zip(wanted, scores, fixed)],
              "folds": folds, "threshold_sweep": sweep, "tta": True,
              "metric": "full-image macro vessel Dice at 512x512, nearest-neighbor labels",
              "selection_note": "Each prediction is from a model not trained on that image, but those images select checkpoints and the shared threshold. This is tuning validation, not an independent test or Kaggle score.",
              "deployment_note": "OOF uses one held-out model per image. Final test predictions average all five models; the transferred threshold has not been independently evaluated for that ensemble."}
    write_json(out / "oof_validation.json", report)
    np.savez_compressed(out / "oof_predictions.npz", ids=np.array(wanted),
                        probabilities=probabilities.astype(np.float32), targets=targets)
    render_training_plots(out, wanted, probabilities, targets, threshold)
    print(json.dumps({k: v for k, v in report.items() if k not in ("threshold_sweep", "per_image")}, indent=2), flush=True)


def render_training_plots(out, ids, probabilities, targets, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for fold in range(5):
        history = out / f"fold_{fold}" / "history.csv"
        if not history.exists():
            continue
        with history.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        axes[0].plot([int(r["epoch"]) for r in rows], [float(r["loss"]) for r in rows], label=f"Fold {fold}")
        measured = [r for r in rows if r["val_dice_0_5"]]
        axes[1].plot([int(r["epoch"]) for r in measured], [float(r["val_dice_0_5"]) for r in measured], label=f"Fold {fold}")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.15)
        ax.legend()
    axes[0].set_ylabel("BCE + soft Dice loss")
    axes[1].set_ylabel("Held-out vessel Dice (threshold 0.5)")
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)
    scores = dice_per_image(probabilities, targets, threshold)
    # Include low/high-scoring images for a less selective visual summary.
    ordered = np.argsort(scores)
    chosen = [ordered[0], ordered[len(ids) // 3], ordered[2 * len(ids) // 3], ordered[-1]]
    fig, axes = plt.subplots(4, 3, figsize=(10, 12), constrained_layout=True)
    for row, i in enumerate(chosen):
        pred = probabilities[i] >= threshold
        disagreement = np.zeros((*pred.shape, 3), dtype=np.float32)
        disagreement[pred & targets[i]] = [0.15, 0.7, 0.3]
        disagreement[pred & ~targets[i]] = [0.95, 0.25, 0.15]
        disagreement[~pred & targets[i]] = [0.15, 0.45, 0.95]
        for ax, panel, title in zip(axes[row], [~targets[i], ~pred, disagreement],
                                   [f"ID {ids[i]}: target (black=vessel)", f"OOF prediction: Dice={scores[i]:.3f}", "Green TP / red FP / blue FN"]):
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.savefig(out / "validation_preview.png", dpi=130)
    plt.close(fig)
