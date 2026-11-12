"""Bounded P1/P2/P3 trials, preserving the original training trajectory."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch

from spatial_models import (ImageBank, INPUT_SIZE, file_hash, make_batch, objective,
                            score, seed_everything, state_hash)
import spatial_models
import next_models

STAGES = {
    "p1_hrnet": dict(model="hrnet", seed=20261022, epochs=80, snapshots=[80], primary=80),
    "p2_rfm": dict(model="rfm", seed=20261021, epochs=40, snapshots=[20,30,40], primary=30),
    "p3_visionfm": dict(model="visionfm", seed=20261021, epochs=40, snapshots=[20,30,40], primary=20),
}


def implementation(name):
    return spatial_models if name == "hrnet" else next_models


def predict(model, *args, **kwargs):
    return implementation(model.encoder.name).predict(model, *args, **kwargs)


def load_saved(path, assets, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    api = implementation(saved["model_name"])
    model = api.SpatialModel(saved["model_name"], assets, saved["seed"], load_pretrained=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    assert state_hash(model.encoder) == saved["frozen_encoder_sha256"]
    return model, saved


def require_gpu():
    assert torch.cuda.is_available(), "CUDA GPU required"


def initialize(seed):
    seed_everything(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def worker(args, smoke=False):
    require_gpu()
    fold = int(args.fold)
    seed = args.seed + fold
    initialize(seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output / args.model / f"fold{fold}"
    if not smoke:
        output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.artifacts / "train.csv")
    test = pd.read_csv(args.artifacts / "test.csv")
    tr = np.flatnonzero(train.fold.values != fold)
    va = np.flatnonzero(train.fold.values == fold)
    assert set(train.iloc[tr].group).isdisjoint(train.iloc[va].group)
    train_bank = ImageBank(train, args.data, "train", device)
    test_bank = ImageBank(test, args.data, "test", device)
    model = implementation(args.model).SpatialModel(args.model, args.assets, seed).to(device)
    source_hash = state_hash(model.encoder)
    initial_head = state_hash(model.head)
    seed_everything(seed)
    torch.set_num_threads(2)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in model.head.parameters())
    if smoke:
        model.train()
        indices = tr[:2]  # Only actual training members of original fold 0.
        images, xy, boxes, theta = make_batch(train_bank, indices, augment=True)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(images, amp=True), xy, boxes, theta, train_bank, indices)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in model.encoder.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
        assert sum(float(p.grad.abs().sum()) for p in model.head.parameters()) > 0
        norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 5.)
        assert torch.isfinite(norm)
        optimizer.step()
        assert state_hash(model.head) != initial_head
        assert state_hash(model.encoder) == source_hash
        result = dict(status="PASS", model_name=args.model, loss=float(loss.detach()),
                      used_training_ids=train.iloc[indices].image_id.tolist(),
                      validation_ids_excluded_from_smoke=train.iloc[va].image_id.tolist(),
                      strict_pretrained_load=model.encoder.load_info, finite_gradients=True,
                      real_head_parameter_update=True, frozen_parameters_and_buffers_unchanged=True,
                      frozen_encoder_sha256=source_hash, head_parameters=trainable,
                      smoke_model_discarded_before_training=True, gpu=torch.cuda.get_device_name(device))
        pd.concat([train_bank.geometry, test_bank.geometry]).to_csv(args.output / "image_geometry.csv", index=False)
        del model, optimizer, train_bank, test_bank
        gc.collect()
        torch.cuda.empty_cache()
        return result

    print(f"START {args.model} fold={fold} train={len(tr)} valid={len(va)} head_params={trainable}", flush=True)
    ema = copy.deepcopy(model.head).eval()
    ema.requires_grad_(False)
    start = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        factor = min(1., epoch / 5) * (.05 + .95 * .5 * (1 + math.cos(
            math.pi * max(0, epoch - 5) / max(1, args.schedule_epochs - 5))))
        optimizer.param_groups[0]["lr"] = args.lr * factor
        losses, norms = [], []
        for indices in np.array_split(np.random.permutation(tr), math.ceil(len(tr) / args.batch_size)):
            images, xy, boxes, theta = make_batch(train_bank, indices, augment=True)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(images, amp=True), xy, boxes, theta, train_bank, indices)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss {args.model} fold={fold} epoch={epoch}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 5.)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite decoder gradient")
            optimizer.step()
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.head.parameters()):
                    averaged.lerp_(current.detach(), .03)
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        record = dict(model_name=args.model, fold=fold, epoch=epoch,
                      loss=float(np.mean(losses)), gradient_norm=float(np.mean(norms)),
                      lr=optimizer.param_groups[0]["lr"], elapsed_seconds=time.perf_counter() - start)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            identity, tta = predict(model, train_bank, va, args.batch_size, head=ema)
            truth = train.iloc[va][["x", "y"]].values
            record.update(val_rmse=score(truth, tta)["rmse_pixels"],
                          identity_val_rmse=score(truth, identity)["rmse_pixels"])
            print(f"{args.model} fold={fold} epoch={epoch:02d}/{args.epochs} "
                  f"loss={record['loss']:.4f} val_RMSE={record['val_rmse']:.3f}px "
                  f"time={record['elapsed_seconds']:.1f}s", flush=True)
        history.append(record)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if epoch in args.snapshots:
            assert state_hash(model.encoder) == source_hash
            snapshot = output / f"epoch{epoch}"
            snapshot.mkdir(exist_ok=True)
            state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            for key, value in ema.state_dict().items():
                state["head." + key] = value.detach().cpu().clone()
            checkpoint = dict(model_name=args.model, fold=fold, seed=seed, completed_epochs=epoch,
                schedule_epochs=args.schedule_epochs, state_dict=state,
                frozen_encoder_sha256=source_hash, initial_head_sha256=initial_head,
                input_size=INPUT_SIZE, source_file_sha256=model.encoder.provenance["sha256"],
                source_load_info=model.encoder.load_info,
                train_ids=train.iloc[tr].image_id.tolist(), validation_ids=train.iloc[va].image_id.tolist())
            torch.save(checkpoint, snapshot / "model.pt")
            manifest = {k:v for k,v in checkpoint.items() if k != "state_dict"}
            manifest.update(checkpoint_sha256=file_hash(snapshot/"model.pt"), head_parameters=trainable,
                actual_head_update=state_hash(model.head)!=initial_head, frozen_encoder_unchanged=True,
                checkpoint_selection="predeclared epoch EMA; snapshot does not mutate live model")
            (snapshot/"training_manifest.json").write_text(json.dumps(manifest,indent=2))
            del state, checkpoint
    assert state_hash(model.encoder) == source_hash
    assert state_hash(model.head) != initial_head
    assert all(p.grad is None for p in model.encoder.parameters())
    del model, ema, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    # Prediction generation begins only after this complete training trajectory.
    for epoch in args.snapshots:
        snapshot = output / f"epoch{epoch}"
        model, saved = load_saved(snapshot / "model.pt", args.assets, device)
        identity, tta = predict(model, train_bank, va, args.batch_size)
        table = train.iloc[va][["image_id","filename","fold","group","width","height","x","y"]].copy()
        table["identity_x"],table["identity_y"] = identity[:,0],identity[:,1]
        table["pred_x"],table["pred_y"] = tta[:,0],tta[:,1]
        table["distance_px"] = np.linalg.norm(table[["x","y"]].values-tta,axis=1)
        table.to_csv(snapshot/"oof.csv",index=False)
        identity, tta = predict(model,test_bank,np.arange(len(test)),args.batch_size)
        table = test.copy()
        table["fold"] = fold
        table["identity_x"],table["identity_y"] = identity[:,0],identity[:,1]
        table["pred_x"],table["pred_y"] = tta[:,0],tta[:,1]
        table.to_csv(snapshot/"test.csv",index=False)
        del model,saved
        gc.collect()
        torch.cuda.empty_cache()
    print(f"COMPLETE {args.model} fold={fold}; snapshots={args.snapshots}",flush=True)


def run(args):
    require_gpu()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(models=[args.model], folds=5, repeats=1, models_trained=5, checkpoints_saved=5*len(args.snapshots), logged_epochs=5*args.epochs, input_size=INPUT_SIZE,
                  encoder_training="fully frozen; eval mode including BN and stochastic depth",
                  decoder="4 x 32-channel projections; shared fusion/refinement; 224x224 heatmap and region",
                  training_precision="frozen encoder fp16 autocast; decoder and losses fp32",
                  evaluation_precision="fp32", tta="identity + horizontal flip, fixed before training",
                  checkpoint_selection="predeclared snapshots; primary epoch in stage protocol",
                  train_manifest_sha256=file_hash(args.artifacts / "train.csv"),
                  torch_version=str(torch.__version__), timm_version=timm.__version__,
                  gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  full_training_on_kaggle=Path("/kaggle/working").is_dir(), kaggle_submission_performed=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    smoke_results = {}
    for name in [args.model]:
        args.model, args.fold, args.device = name, 0, "cuda:0"
        smoke_results[name] = worker(args, smoke=True)
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU PREFLIGHT PASS:", name, flush=True)
    (args.output / "gpu_smoke_checks.json").write_text(json.dumps(smoke_results, indent=2))
    jobs = [(name, fold) for fold in range(5) for name in [args.model]]
    devices = list(range(min(2, torch.cuda.device_count())))
    active = {}
    while jobs or active:
        for gpu in devices:
            if gpu in active or not jobs:
                continue
            name, fold = jobs.pop(0)
            folder = args.output / name / f"fold{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            log_path = folder / "worker.log"
            log_handle = log_path.open("w", buffering=1)
            command = [sys.executable, "-u", __file__, "--worker", "--model", name,
                       "--fold", str(fold), "--device", f"cuda:{gpu}",
                       "--data", str(args.data), "--artifacts", str(args.artifacts),
                       "--assets", str(args.assets), "--output", str(args.output),
                       "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                       "--lr", str(args.lr), "--seed", str(args.seed),
                       "--stage",args.stage,"--schedule-epochs",str(args.schedule_epochs)]
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
            active[gpu] = dict(process=process, log=log_handle, path=log_path, offset=0, name=name, fold=fold)
            print(f"Launched {name} fold={fold} on cuda:{gpu}", flush=True)
        for gpu, item in list(active.items()):
            with item["path"].open() as stream:
                stream.seek(item["offset"])
                new = stream.read()
                item["offset"] = stream.tell()
            if new:
                print(new, end="", flush=True)
            result = item["process"].poll()
            if result is not None:
                item["log"].close()
                if result:
                    for other in active.values():
                        if other["process"].poll() is None:
                            other["process"].terminate()
                    raise RuntimeError(f"Worker failed: {item['name']} fold={item['fold']}; see {item['path']}")
                del active[gpu]
        if active:
            time.sleep(3)
    print(f"TRAINING COMPLETE: {args.stage}, 5 trajectories, {5*len(args.snapshots)} snapshots.",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("next_run"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=["hrnet","rfm","visionfm"])
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--schedule-epochs",type=int,default=80)
    parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = STAGES[args.stage]
    args.model,args.seed,args.epochs,args.snapshots = protocol["model"],protocol["seed"],protocol["epochs"],protocol["snapshots"]
    assert args.schedule_epochs == 80
    if args.smoke_only:
        args.output.mkdir(parents=True,exist_ok=True)
        args.fold=0
        print(json.dumps(worker(args,smoke=True),indent=2))
    else:
        worker(args) if args.worker else run(args)
