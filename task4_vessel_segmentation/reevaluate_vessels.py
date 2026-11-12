"""Re-evaluate the same five held-out checkpoints with explicit projection/TTA."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from retinal import write_json
from vessel_model import get_split
from strong_vessels import finalize

ROOT = Path(__file__).resolve().parent


def source_checkpoint(source, source_name, fold):
    matches = []
    for path in Path(source).rglob(f"fold_{fold}/best.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["config"]["name"] == source_name and int(checkpoint["fold"]) == fold:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Expected one exact source checkpoint for fold {fold}; found {len(matches)}")
    return matches[0]


def run_fold(config, checkpoint_path, data, out, fold, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_ids, val_ids = get_split(data, fold, 42)
    split = checkpoint["split"]
    if split["train_ids"] != train_ids or split["val_ids"] != val_ids or set(train_ids) & set(val_ids):
        raise ValueError("Source checkpoint changed the fixed held-out split")
    if checkpoint["config"]["encoder"] != config["encoder"]:
        raise ValueError("Unexpected source model architecture")
    directory = Path(out) / f"fold_{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "report.json").exists():
        raise FileExistsError("Preserve existing evaluation results")
    command = [sys.executable, "-u", str(ROOT / "strong_vessels.py"), "evaluate",
               "--checkpoint", str(checkpoint_path), "--data", str(data), "--out", str(directory),
               "--device", device, "--inference-mode", config["inference_mode"],
               "--inference-patch", str(config["inference_patch"]),
               "--projection", config["inference_projection"], "--tta", str(config["tta"])]
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    report.update(name=config["name"], best_epoch=checkpoint["epoch"],
                  pretrained_source=checkpoint["pretrained_source"],
                  source_checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                  scope="re-evaluation of unchanged held-out weights; no current-label fitting")
    report["config"].update(name=config["name"], tta=config["tta"], source_name=config["source_name"], task="reevaluate")
    checkpoint["config"] = report["config"]
    checkpoint["source_checkpoint_sha256"] = report["source_checkpoint_sha256"]
    torch.save(checkpoint, directory / "best.pt")
    write_json(directory / "report.json", report)
    write_json(directory / "split.json", split)
    write_json(directory / "config.json", report["config"])
    print("REEVALUATE_FOLD_COMPLETE", fold, report["mean_vessel_dice"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/segmentation")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fold", choices=["all", "0", "1", "2", "3", "4"], default="all")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folds = list(range(5)) if args.fold == "all" else [int(args.fold)]
    paths = {fold: source_checkpoint(args.source, config["source_name"], fold) for fold in folds}
    # The input kernel already carries hash-checked public model assets.
    cache_source = paths[folds[0]].parents[2] / "task4_source/optimization/thirdparty"
    cache_target = ROOT / "optimization/thirdparty"
    if cache_source.is_dir() and cache_source.resolve() != cache_target.resolve():
        shutil.copytree(cache_source, cache_target, dirs_exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for checkpoint re-evaluation")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", config)
    devices = [f"cuda:{i}" for i in range(min(torch.cuda.device_count(), 2))]
    witness = {"torch": str(torch.__version__), "devices": []}
    for device in devices:
        matrix = torch.randn(32, 32, device=device)
        assert torch.isfinite(matrix @ matrix).all()
        witness["devices"].append({"device": device, "name": torch.cuda.get_device_name(device), "kernel_witness": "passed"})
    write_json(out / "environment_witness.json", witness)

    def worker(index):
        for fold in folds[index::len(devices)]:
            run_fold(config, paths[fold], args.data, out, fold, devices[index])

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, i) for i in range(len(devices))]
        for future in futures:
            future.result()
    if args.fold == "all":
        report = finalize(args.data, out)
        report.update(scope="fixed 20-image five-fold OOF re-evaluation of the same trained checkpoints",
                      source_name=config["source_name"], source_kernels=config.get("kernel_sources", []))
        write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
