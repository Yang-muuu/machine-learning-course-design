"""固定分组折上的训练、诊断与预测；不接收答案文件。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split

from cnn_models import CNN, Config, file_hash, make_optimizer, single_loss, update_ema
from model_and_aug import CLASS_NAMES, EmotionDataset, make_loader, make_transform, mix_batch, seed_everything


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def metrics(labels, probabilities):
    assert probabilities.shape == (len(labels), 6)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert np.allclose(probabilities.sum(1), 1, atol=1e-5)
    predicted = probabilities.argmax(1)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, labels=range(6), average="macro", zero_division=0)),
            "log_loss": float(log_loss(labels, probabilities, labels=list(range(6))))}


def train_step(model, ema, images, labels, optimizer, scaler, config, device, epoch, forced=None):
    images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
    labels = labels.to(device, non_blocking=True)
    mixed = epoch <= config.epochs - config.unmixed_final_epochs
    if mixed:
        images, first, second, ratio, kind = mix_batch(images, labels, config, forced=forced)
    else:
        first, second, ratio, kind = labels, labels, 1.0, "none"
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(images)
        loss = ratio * single_loss(logits, first, config, epoch) + (1 - ratio) * single_loss(logits, second, config, epoch)
        supervised = loss
        auxiliary = model.pop_auxiliary_loss()
        if auxiliary is not None:
            # 对混合标签的CE已经正确加权；与标签无关的注意力辅助项只加一次。
            loss = loss + auxiliary
    if not torch.isfinite(loss):
        raise FloatingPointError("训练损失出现 NaN/Inf")
    model.last_supervised_loss = float(supervised.detach())
    model.last_auxiliary_loss = float(auxiliary.detach()) if auxiliary is not None else 0.0
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    previous = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    model.last_optimizer_step_skipped = scaler.get_scale() < previous
    if ema is not None and scaler.get_scale() >= previous:
        update_ema(ema, model, 0.0 if epoch <= config.ema_warmup_epochs else config.ema_decay)
    predicted = logits.detach().argmax(1)
    correct = ratio * (predicted == first).float().sum() + (1 - ratio) * (predicted == second).float().sum()
    return float(loss.detach()), float(correct), kind


@torch.inference_mode()
def predict(model, loader, device, progress=None):
    model.eval()
    batches = []
    for batch, (images, _labels) in enumerate(loader):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probabilities = (model(images).float().softmax(1) + model(images.flip(-1)).float().softmax(1)) / 2
        batches.append(probabilities.cpu().numpy())
        if progress is not None:
            progress(batch)
    return np.concatenate(batches).astype(np.float32)


def fit_fold(context, config, fold, gpu_rank):
    config.validate()
    target = Path(context["output"]) / "runs" / f"fold_{fold}"
    target.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed + fold, config.cpu_threads)
    device = torch.device(context.get("device", "cuda:0"))
    cache = Path(context["cache"])
    split = pd.read_csv(context["split"], dtype={"ID": str})
    manifest = pd.read_csv(cache / "train_manifest.csv", dtype={"ID": str})
    labels = manifest.label.to_numpy(dtype=np.int64)
    fitting = split.loc[split.validation_fold != fold]
    validation = split.loc[split.validation_fold == fold]
    if context.get("smoke"):
        fitting = fitting.groupby("label", group_keys=False).head(2)
        validation = validation.groupby("label", group_keys=False).head(1)
    assert set(fitting.label) == set(validation.label) == set(range(6))
    assert not set(fitting.pixel_hash) & set(validation.pixel_hash)
    train_dataset = EmotionDataset(cache / "train_images.npy", fitting.array_index.to_numpy(), labels, make_transform(config, True))
    val_dataset = EmotionDataset(cache / "train_images.npy", validation.array_index.to_numpy(), labels, make_transform(config, False))
    # 干净训练子集与验证集采用相同预处理、EMA 和翻转预测，才能诊断泛化差距。
    if len(fitting) <= config.clean_train_samples:
        clean_rows = fitting.copy()
    else:
        clean_rows, _ = train_test_split(fitting, train_size=config.clean_train_samples,
                                        stratify=fitting.label, random_state=config.seed)
    clean_rows[["ID", "label", "array_index"]].to_csv(target / "clean_train_rows.csv", index=False)
    clean_dataset = EmotionDataset(cache / "train_images.npy", clean_rows.array_index.to_numpy(), labels, make_transform(config, False))
    train_loader = make_loader(train_dataset, config, True, device, config.seed + fold)
    val_loader = make_loader(val_dataset, config, False, device, config.seed + fold)
    clean_loader = make_loader(clean_dataset, config, False, device, config.seed + fold)
    model = CNN(config, context.get("weights")).to(device, memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False) if config.ema_decay else None
    optimizer = make_optimizer(model, config)

    def rate(epoch):
        warmup = min(config.warmup_epochs, max(0, config.epochs - 1))
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, config.epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    # 极小冒烟只有两个batch，允许较低初始缩放；正式训练保持默认65536及实际更新检查。
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda",
                                init_scale=4096.0 if context.get("smoke") else 65536.0)
    history, best, best_probability = [], None, None
    start, no_improvement, counts = time.monotonic(), 0, Counter()
    amp_skipped_steps = 0
    stop_reason = "max_epochs"
    write_json(target / "config.json", {**config.dictionary(), "fold": fold, "gpu_rank": gpu_rank})
    print(f"PHASE fold_start name={context['name']} fold={fold} gpu={gpu_rank} train={len(fitting)} val={len(validation)}", flush=True)
    def stage(phase, current_epoch=0):
        last = [0.0]
        def progress(batch):
            if batch == 0 or time.time() - last[0] >= 10:
                write_json(target / "heartbeat.json", {"wall": time.time(), "epoch": current_epoch,
                           "phase": phase, "batch": batch})
                last[0] = time.time()
            if time.time() >= context["deadline"]:
                raise TimeoutError("评估到达全局截止时间")
        progress(0)
        print(f"PHASE evaluation name={context['name']} fold={fold} epoch={current_epoch} stage={phase}", flush=True)
        return progress
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        loss_total, correct_total, count = 0.0, 0.0, 0
        supervised_total, auxiliary_total = 0.0, 0.0
        successful_updates = 0
        for batch, (images, targets) in enumerate(train_loader):
            loss, correct, kind = train_step(model, ema, images, targets, optimizer, scaler, config, device, epoch)
            loss_total += loss * len(images)
            supervised_total += model.last_supervised_loss * len(images)
            auxiliary_total += model.last_auxiliary_loss * len(images)
            correct_total += correct
            count += len(images)
            counts[kind] += 1
            amp_skipped_steps += int(model.last_optimizer_step_skipped)
            successful_updates += int(not model.last_optimizer_step_skipped)
            if batch % 60 == 0:
                write_json(target / "heartbeat.json", {"wall": time.time(), "epoch": epoch, "batch": batch})
                print(f"HEARTBEAT name={context['name']} fold={fold} epoch={epoch} batch={batch}/{len(train_loader)}", flush=True)
            if time.time() >= context["deadline"]:
                raise TimeoutError("训练到达全局截止时间，保留已完成结果但不声称全部完成")
        assert successful_updates > 0, "本轮 AMP 没有实际更新参数，停止而不是报告训练完成"
        ordinary_probability = predict(model, val_loader, device, stage("validation_ordinary", epoch))
        ordinary_score = metrics(validation.label.to_numpy(), ordinary_probability)
        evaluated = ema if config.selection == "ema" and ema is not None else model
        probability = predict(evaluated, val_loader, device, stage("validation_ema", epoch)) if evaluated is not model else ordinary_probability
        score = metrics(validation.label.to_numpy(), probability)
        clean_score = metrics(clean_rows.label.to_numpy(), predict(evaluated, clean_loader, device, stage("clean_train", epoch)))
        record = {"epoch": epoch, "train_loss": loss_total / count, "mixed_train_accuracy": correct_total / count,
                  "supervised_train_loss": supervised_total / count, "attention_auxiliary_loss": auxiliary_total / count,
                  **{"val_" + key: value for key, value in score.items()},
                  **{"ordinary_val_" + key: value for key, value in ordinary_score.items()},
                  "clean_train_accuracy": clean_score["accuracy"], "clean_train_log_loss": clean_score["log_loss"],
                  "generalization_gap": clean_score["accuracy"] - score["accuracy"],
                  "ema_minus_ordinary_accuracy": score["accuracy"] - ordinary_score["accuracy"],
                  "backbone_lr": next(group["lr"] for group in optimizer.param_groups if group["group_kind"]=="backbone"),
                  "head_lr": next(group["lr"] for group in optimizer.param_groups if group["group_kind"]=="head"),
                  "seconds": time.monotonic() - epoch_start,
                  "amp_skipped_steps_cumulative": amp_skipped_steps,
                  "successful_optimizer_updates_this_epoch": successful_updates}
        history.append(record)
        key = (score["accuracy"], -score["log_loss"])
        if best is None or key > (best["accuracy"], -best["log_loss"]):
            best = {**score, "best_epoch": epoch}
            best_probability = probability.copy()
            no_improvement = 0
            state = {"model": {name: value.detach().cpu() for name, value in evaluated.state_dict().items()},
                     "config": config.dictionary(), "fold": fold, "epoch": epoch,
                     "evaluation_model": config.selection, "smoke": bool(context.get("smoke")),
                     "source_hashes": context.get("source_hashes", {}),
                     "pretrained_manifest": context.get("pretrained_manifest", {}),
                     "torch_version": str(torch.__version__)}
            torch.save(state, target / "best.tmp")
            (target / "best.tmp").replace(target / "best.pt")
        else:
            no_improvement += 1
        scheduler.step()
        pd.DataFrame(history).to_csv(target / "history.csv", index=False)
        write_json(target / "progress.json", {"fold": fold, "epoch": epoch, "best": best, "wall": time.time()})
        print(f"RESULT name={context['name']} fold={fold} epoch={epoch} val={score['accuracy']:.5f} raw={ordinary_score['accuracy']:.5f} clean_train={clean_score['accuracy']:.5f} seconds={record['seconds']:.1f}", flush=True)
        if config.patience and epoch >= config.min_epochs and no_improvement >= config.patience:
            stop_reason = "early_stopping"
            break
    assert best_probability is not None
    saved = torch.load(target / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    replay = predict(model, val_loader, device, stage("checkpoint_replay"))
    difference = float(np.abs(replay - best_probability).max())
    assert difference <= 1e-5, "最佳权重重载后预测不一致"
    rows = validation[["ID", "label", "pixel_hash", "array_index"]].copy()
    rows["prediction"] = best_probability.argmax(1)
    rows.to_csv(target / "validation_rows.csv", index=False)
    np.save(target / "validation_probabilities.npy", best_probability)
    test_manifest = pd.read_csv(cache / "test_manifest.csv", dtype={"ID": str})
    test_positions = np.arange(min(8, len(test_manifest)) if context.get("smoke") else len(test_manifest))
    test_dataset = EmotionDataset(cache / "test_images.npy", test_positions, None, make_transform(config, False))
    test_loader = make_loader(test_dataset, config, False, device, config.seed + fold)
    test_probability = predict(model, test_loader, device, stage("test_prediction"))
    np.save(target / "test_probabilities.npy", test_probability)
    report = classification_report(validation.label, best_probability.argmax(1), labels=range(6), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    write_json(target / "classification_report.json", report)
    write_json(target / "confusion_matrix.json", confusion_matrix(validation.label, best_probability.argmax(1), labels=range(6)).tolist())
    best_record = history[best["best_epoch"] - 1]
    reflection = {
        "best_epoch": best["best_epoch"], "last_accuracy": history[-1]["val_accuracy"],
        "post_best_change": history[-1]["val_accuracy"] - best["accuracy"],
        "clean_train_validation_gap": best_record["generalization_gap"],
        "ema_minus_ordinary_at_selected_epoch": best_record["ema_minus_ordinary_accuracy"],
        "fear_recall": report["Fear"]["recall"], "sad_recall": report["Sad"]["recall"],
        "interpretation": "用相同无增强预处理比较训练子集与验证集；难样本和错标需区分，不能由大损失自动判定错标。",
    }
    write_json(target / "reflection.json", reflection)
    summary = {"name": context["name"], "fold": fold, **best, "epochs_completed": len(history),
               "validation_count": len(validation), "test_count": len(test_positions), "gpu_rank": gpu_rank,
               "stop_reason": stop_reason, "seconds": time.monotonic() - start, "mixing_counts": dict(counts),
               "amp_skipped_steps": amp_skipped_steps,
               "checkpoint_sha256": file_hash(target / "best.pt"), "reload_max_difference": difference,
               "architecture": config.architecture, "head": config.head, "attention": config.attention,
               "native_attention": bool(getattr(model, "native_attention", False)),
               "additional_attention": config.additional_attention,
               "attention_diversity_weight": config.attention_diversity_weight,
               "source_hashes": context.get("source_hashes", {}),
               "pretrained_manifest": context.get("pretrained_manifest", {}),
               "loss_kind": config.loss_kind, "reflection": reflection, "smoke": bool(context.get("smoke"))}
    write_json(target / "run_summary.json", summary)
    print(f"PHASE fold_done name={context['name']} fold={fold} accuracy={best['accuracy']:.5f}", flush=True)
    return summary


def probe(context, config, rank):
    config.validate()
    seed_everything(config.seed + rank, config.cpu_threads)
    device = torch.device(context.get("device", "cuda:0"))
    cache = Path(context["cache"])
    labels = pd.read_csv(cache / "train_manifest.csv").label.to_numpy(dtype=np.int64)
    positions = np.concatenate([np.flatnonzero(labels == label)[:2] for label in range(6)])
    dataset = EmotionDataset(cache / "train_images.npy", positions, labels, make_transform(config, True))
    loader = make_loader(dataset, config, True, device, config.seed)
    images, targets = next(iter(loader))
    model = CNN(config, context.get("weights")).to(device, memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    losses = []
    before = model.classifier[-1].weight.detach().clone()
    skipped_steps = 0
    start = time.monotonic()
    for kind in ["none", "mixup", "cutmix"]:
        model.train()
        # 初始 GradScaler 可能需要回退；每种混合方式都必须完成一次真实更新。
        for _attempt in range(8):
            loss, _, actual = train_step(model, ema, images, targets, optimizer, scaler, config, device, 1, forced=kind)
            if not model.last_optimizer_step_skipped:
                break
            skipped_steps += 1
        else:
            raise FloatingPointError(f"{kind} 连续八次 AMP 更新被跳过")
        assert actual == kind
        losses.append(loss)
    evaluation_dataset = EmotionDataset(cache / "train_images.npy", positions, labels, make_transform(config, False))
    evaluation_loader = make_loader(evaluation_dataset, config, False, device, config.seed)
    first = predict(ema, evaluation_loader, device)
    model.load_state_dict(ema.state_dict(), strict=True)
    replay = predict(model, evaluation_loader, device)
    difference = float(np.abs(first - replay).max())
    parameter_change = float((model.classifier[-1].weight.detach() - before).abs().max())
    assert difference <= 1e-5 and np.isfinite(losses).all()
    assert parameter_change > 0, "优化器未改变参数"
    result = {"rank": rank, "architecture": config.architecture, "head": config.head, "attention": config.attention,
              "native_attention": bool(getattr(model, "native_attention", False)),
              "successful_optimizer_updates": 3, "amp_skipped_steps": skipped_steps,
              "head_parameter_max_change": parameter_change,
              "losses": losses, "reload_max_difference": difference, "seconds": time.monotonic() - start,
              "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "passed": True}
    write_json(Path(context["output"]) / f"probe_{rank}.json", result)
    print(f"PHASE probe_passed rank={rank} result={result}", flush=True)


def profile(context, config, rank):
    """以正式批大小做短测速；包含数据加载，不把纯 GPU 算子速度当作训练速度。"""
    config.validate()
    seed_everything(config.seed + rank, config.cpu_threads)
    device = torch.device(context.get("device", "cuda:0"))
    assert device.type == "cuda", "完整训练耗时必须在目标 GPU 上测量"
    cache = Path(context["cache"])
    labels = pd.read_csv(cache / "train_manifest.csv").label.to_numpy(dtype=np.int64)
    train_positions = np.arange(config.batch_size * 10)
    train_dataset = EmotionDataset(cache / "train_images.npy", train_positions, labels, make_transform(config, True))
    train_loader = make_loader(train_dataset, config, True, device, config.seed)
    val_positions = np.arange(config.eval_batch_size * 4)
    val_dataset = EmotionDataset(cache / "train_images.npy", val_positions, labels, make_transform(config, False))
    val_loader = make_loader(val_dataset, config, False, device, config.seed)
    model = CNN(config, context.get("weights")).to(device, memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(train_loader)
    model.train()
    print(f"PHASE profile_warmup architecture={config.architecture} batch={config.batch_size} rank={rank}", flush=True)
    warmup_updates, warmup_steps = 0, 0
    while warmup_updates < 2 and warmup_steps < 10:
        images, targets = next(iterator)
        train_step(model, ema, images, targets, optimizer, scaler, config, device, 1, forced="none")
        warmup_updates += int(not model.last_optimizer_step_skipped)
        warmup_steps += 1
    assert warmup_updates == 2, "AMP 预热无法实际更新参数"
    torch.cuda.synchronize()
    # 重新创建迭代器，避免 AMP 回退较多时提前耗尽预检的小数据集。
    iterator = iter(train_loader)
    start, measured_steps, measured_updates = time.monotonic(), 6, 0
    for _ in range(measured_steps):
        images, targets = next(iterator)
        train_step(model, ema, images, targets, optimizer, scaler, config, device, 1, forced="none")
        measured_updates += int(not model.last_optimizer_step_skipped)
    torch.cuda.synchronize()
    train_step_seconds = (time.monotonic() - start) / measured_steps
    assert measured_updates >= 5, "测速阶段 AMP 更新跳过过多，不能用该吞吐估计训练"
    print(f"PHASE profile_training_ready architecture={config.architecture} seconds_per_step={train_step_seconds:.4f}", flush=True)
    # 先预热验证批大小；正式计时仍包含加载和原图/翻转两次预测。
    images, _ = next(iter(val_loader))
    images = images.to(device, memory_format=torch.channels_last)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        ema(images)
        ema(images.flip(-1))
    torch.cuda.synchronize()
    start = time.monotonic()
    probability = predict(ema, val_loader, device)
    torch.cuda.synchronize()
    eval_seconds_per_image = (time.monotonic() - start) / len(val_dataset)
    assert probability.shape == (len(val_dataset), 6) and np.isfinite(probability).all()
    record = {"architecture": config.architecture, "rank": rank, "passed": True,
              "gpu": torch.cuda.get_device_name(0), "batch_size": config.batch_size,
              "eval_batch_size": config.eval_batch_size, "training_steps_measured": measured_steps,
              "successful_measured_updates": measured_updates, "warmup_steps": warmup_steps,
              "train_step_seconds": train_step_seconds, "eval_seconds_per_image_with_flip": eval_seconds_per_image,
              "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
              "note": "短样本测速包含加载；耗时估计仍有温度、CPU 争用、平台排队等误差。"}
    write_json(Path(context["output"]) / f"profile_{rank}.json", record)
    print(f"PHASE profile_passed rank={rank} result={record}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text(encoding="utf-8"))
    config = Config(**json.loads(Path(args.config).read_text(encoding="utf-8"))).validate()
    assert not (args.probe and args.profile)
    if args.profile:
        profile(context, config, args.rank)
    elif args.probe:
        probe(context, config, args.rank)
    else:
        fit_fold(context, config, args.fold, args.rank)


if __name__ == "__main__":
    main()
