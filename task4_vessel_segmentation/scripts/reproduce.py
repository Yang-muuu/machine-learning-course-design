"""Replay the final five-branch submission or check its frozen validation scores."""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from competition_protocol import binary_predictions, dice, read_targets
from ensemble_recipe import replay_recipe
from native_rasterization import confidence_field
from retinal import index_images, template_ids, write_json
from submission import verify_submission, write_submission


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_scores(path, ids, key="probabilities"):
    with np.load(path, allow_pickle=False) as saved:
        if saved["ids"].tolist() != ids:
            raise ValueError(f"Unexpected image order in {path}")
        values = saved[key].astype(np.float32)
    if values.shape != (len(ids), 512, 512) or not np.isfinite(values).all():
        raise ValueError(f"Unexpected score-map shape or values in {path}")
    if values.min() < 0 or values.max() > 1:
        raise ValueError(f"Scores must lie in [0, 1] in {path}")
    return values


def member_path(cache, alias, slug):
    canonical = cache / f"{alias}.npz"
    if canonical.is_file():
        return canonical
    matches = list(cache.glob(f"*__{slug}.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected {canonical.name} or one *__{slug}.npz in {cache}")
    return matches[0]


def blend_from_cache(cache, ids, frozen):
    probabilities = {
        alias: load_scores(member_path(cache, alias, branch["cache_slug"]), ids)
        for alias, branch in frozen["members"].items()
    }
    return replay_recipe(frozen["recipe"], probabilities)


def export(args, frozen):
    ids = template_ids(args.data)
    blend = blend_from_cache(args.cache, ids, frozen)
    masks = np.where(binary_predictions(blend, frozen["threshold"], frozen["postprocess"]), 0, 255).astype(np.uint8)
    mask_dir = args.out / "test_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for key, mask in zip(ids, masks):
        Image.fromarray(mask).save(mask_dir / f"{key}.png")
    csv_path = args.out / "submission.csv"
    write_submission(csv_path, ids, masks)
    report = verify_submission(args.data, csv_path, mask_dir)
    report["submission_sha256"] = digest(csv_path)
    report["matches_frozen_submission"] = report["submission_sha256"] == frozen["submission_sha256"]
    if args.check_converter:
        converter = args.data / "segmentation_to_csv.py"
        if digest(converter) != frozen["converter_sha256"]:
            raise ValueError("Converter SHA-256 differs from the original task attachment")
        original = runpy.run_path(str(converter))["turn_to_str"]
        with tempfile.TemporaryDirectory(prefix="white_vessel_", dir=args.out) as temporary:
            paths = []
            for key, mask in zip(ids, masks):
                path = Path(temporary) / f"{key}.png"
                Image.fromarray(255 - mask).save(path)
                paths.append(str(path))
            with csv_path.open(newline="", encoding="utf-8") as stream:
                expected = [row["Predicted"] for row in csv.DictReader(stream)]
            if original(paths) != expected:
                raise ValueError("Original converter output differs for vessel-white input")
        report["official_comparison"] = "all rows match the supplied converter with vessel-white input"
    write_json(args.out / "submission_check.json", report)
    print(json.dumps({key: report[key] for key in ("rows", "matches_frozen_submission",
          "submission_sha256", "official_comparison")}, indent=2))
    if args.exact and not report["matches_frozen_submission"]:
        raise ValueError("Exported CSV differs from the frozen submission")


def score_oof(args, frozen):
    ids = [str(i) for i in range(1, 21)]
    labels = index_images(args.data / "train/label")
    for key in ids:
        if digest(labels[key]) != frozen["label_sha256"][key]:
            raise ValueError(f"Label {key} differs from the original task attachment")
    targets = read_targets(args.data, ids)
    if args.predictions is not None:
        probabilities = load_scores(args.predictions, ids)
        with np.load(args.predictions, allow_pickle=False) as saved:
            if not np.array_equal(saved["targets"], targets):
                raise ValueError("Saved OOF targets differ from freshly processed original labels")
    else:
        probabilities = blend_from_cache(args.cache, ids, frozen)
    prediction = binary_predictions(probabilities, frozen["threshold"], frozen["postprocess"])
    values = dice(prediction, targets)
    matches = np.allclose(values, [frozen["per_image_dice"][key] for key in ids], atol=1e-12, rtol=0)
    if args.predictions is not None or args.exact:
        if not matches:
            raise ValueError("Per-image Dice differs from the frozen report")
        record = json.loads((ROOT / "records/oof_counts.json").read_text(encoding="utf-8"))
        for key, predicted, target in zip(ids, prediction, targets):
            counts = {"tp": int((predicted & target).sum()),
                      "fp": int((predicted & ~target).sum()),
                      "fn": int((~predicted & target).sum())}
            if counts != record["per_image"][key]:
                raise ValueError(f"Per-image pixel counts differ from the frozen report for ID {key}")
    print(json.dumps({"images": len(ids), "oof_macro_dice": float(values.mean()),
                      "matches_frozen_dice": bool(matches), "per_image_dice": dict(zip(ids, values.tolist())),
                      "scope": "fixed fivefold tuning validation"}, indent=2))


def teacher_maps(args):
    from vessel_model import read_label, setup
    from vascx_adapter import (VascXAdapter, prepare_image, predict_author_pipeline,
                               provenance, restore_probability)

    folder = args.data / ("train/image" if args.split == "train" else "test/image")
    paths = index_images(folder)
    ids = [str(i) for i in range(1, 21)] if args.split == "train" else template_ids(args.data)
    device = setup(42, args.device)
    model = VascXAdapter({"vascx_member": "all"}).eval().requires_grad_(False).to(device)
    fields = []
    for key in ids:
        image, geometry = prepare_image(str(paths[key]), 1024)
        fields.append(restore_probability(predict_author_pipeline(model, image, device), geometry))
        print(f"VascX {args.split} ID {key}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ids": np.array(ids)}
    key = "probabilities" if args.split == "train" else "direct_probabilities"
    payload[key] = np.stack(fields).astype(np.float32)
    if args.split == "train":
        labels = index_images(args.data / "train/label")
        payload["targets"] = np.stack([read_label(labels[key], 512) for key in ids]).astype(bool)
        write_json(args.out.with_name("report.json"), {
            "scope": "external-only reference; frozen feature extractor",
            "pretraining": provenance(),
        })
    np.savez_compressed(args.out, **payload)


def predict_neural(args, frozen, alias, ids, paths):
    import torch
    from strong_vessels import cv_preprocess, fov512, make_model, predict, prediction_geometries
    from vessel_model import setup

    branch = frozen["members"][alias]
    device = setup(42, args.device)
    torch.set_num_threads(2)
    totals = np.zeros((len(ids), 512, 512), dtype=np.float32)
    indices = {key: index for index, key in enumerate(ids)}
    fovs = {key: fov512(paths[key]) for key in ids}
    for fold, split in enumerate(frozen["splits"]):
        path = args.model_dir / branch["weights_folder"] / "results" / f"fold_{fold}/best.pt"
        if args.exact_weights and digest(path) != branch["checkpoint_sha256"][fold]:
            raise ValueError(f"Unexpected checkpoint contents in {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = dict(checkpoint["config"])
        if (config["name"] != branch["config_name"] or int(checkpoint["fold"]) != fold
                or any(checkpoint["split"][key] != split[key] for key in ("train_ids", "val_ids"))):
            raise ValueError(f"Unexpected model or split in {path}")
        model = make_model(config, pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model = model.eval().requires_grad_(False).to(device, memory_format=torch.channels_last)
        predict_ids = split["val_ids"] if args.split == "oof" else ids
        geometries = prediction_geometries(paths, predict_ids, config)
        for key, geometry in zip(predict_ids, geometries):
            if alias == "fr96":
                geometry = dict(geometry, return_native=True)
            field = predict(model, cv_preprocess(paths[key], config), device, config.get("tta", 4), geometry)
            field = confidence_field(field, fovs[key]) if alias == "fr96" else field * fovs[key]
            if args.split == "oof":
                totals[indices[key]] = field
            else:
                totals[indices[key]] += field / np.float32(5)
        print(f"{alias} fold {fold} finished", flush=True)
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return totals


def predict_pixels(args, frozen, ids, paths):
    import joblib
    from threadpoolctl import threadpool_limits
    from pixel_vessels import features
    from scripts.predict_binned import predict_proba_binned
    from strong_vessels import fov512

    teacher_key = "probabilities" if args.split == "oof" else "direct_probabilities"
    teachers = load_scores(args.teacher, ids, teacher_key)
    branch = frozen["members"]["pixel"]
    models = []
    for fold, split in enumerate(frozen["splits"]):
        directory = args.pixel_dir / f"fold_{fold}"
        stored = json.loads((directory / "split.json").read_text(encoding="utf-8"))
        if any(stored[key] != split[key] for key in ("train_ids", "val_ids")):
            raise ValueError(f"Unexpected pixel model split in {directory}")
        path = directory / "model.joblib"
        if args.exact_weights and digest(path) != branch["checkpoint_sha256"][fold]:
            raise ValueError(f"Unexpected pixel model contents in {path}")
        model = joblib.load(path)
        model._bin_mapper.n_threads = 2
        models.append(model)
    totals = np.zeros((len(ids), 512, 512), dtype=np.float32)
    fold_by_id = {key: fold for fold, split in enumerate(frozen["splits"]) for key in split["val_ids"]}
    with threadpool_limits(limits=2):
        for index, key in enumerate(ids):
            x, names = features(paths[key], teachers[index], native=False)
            if names != branch["feature_names"] or x.shape != (512 * 512, 71):
                raise ValueError("Pixel features differ from those used to fit the models")
            selected = [models[fold_by_id[key]]] if args.split == "oof" else models
            for model in selected:
                values = predict_proba_binned(model, x, threads=2)[:, 1].reshape(512, 512)
                totals[index] += values.astype(np.float32) / np.float32(len(selected))
            totals[index] *= fov512(paths[key])
            print(f"Pixel HGB ID {key} finished", flush=True)
    return totals


def predict_members(args, frozen):
    ids = [str(i) for i in range(1, 21)] if args.split == "oof" else template_ids(args.data)
    folder = "train/image" if args.split == "oof" else "test/image"
    paths = index_images(args.data / folder)
    if set(paths) != set(ids):
        raise ValueError("Image IDs differ from the fixed dataset")
    members = list(frozen["members"]) if args.member == "all" else [args.member]
    args.out.mkdir(parents=True, exist_ok=True)
    for alias in members:
        target = args.out / f"{alias}.npz"
        if target.exists():
            raise FileExistsError(f"Use a new output directory to preserve {target}")
        if alias == "pixel":
            values = predict_pixels(args, frozen, ids, paths)
        else:
            values = predict_neural(args, frozen, alias, ids, paths)
        pending = target.with_suffix(".next.npz")
        np.savez_compressed(pending, ids=np.array(ids), probabilities=values)
        pending.replace(target)


def main():
    frozen = json.loads((ROOT / "records/final_snapshot.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "score-oof", "teacher", "predict"):
        commands.add_parser(name).add_argument("--data", type=Path, default=Path("data/segmentation"))
    export_cli = commands.choices["export"]
    export_cli.add_argument("--cache", type=Path, required=True)
    export_cli.add_argument("--out", type=Path, default=Path("runs/export"))
    export_cli.add_argument("--check-converter", action="store_true")
    export_cli.add_argument("--exact", action="store_true")
    oof_cli = commands.choices["score-oof"]
    oof_input = oof_cli.add_mutually_exclusive_group(required=True)
    oof_input.add_argument("--predictions", type=Path)
    oof_input.add_argument("--cache", type=Path)
    oof_cli.add_argument("--exact", action="store_true")
    teacher_cli = commands.choices["teacher"]
    teacher_cli.add_argument("--split", choices=["train", "test"], required=True)
    teacher_cli.add_argument("--device", default="cuda:0")
    teacher_cli.add_argument("--out", type=Path, required=True)
    predict_cli = commands.choices["predict"]
    predict_cli.add_argument("--member", choices=["all", *frozen["members"]], default="all")
    predict_cli.add_argument("--split", choices=["test", "oof"], default="test")
    predict_cli.add_argument("--device", default="cuda:0")
    predict_cli.add_argument("--model-dir", type=Path, default=Path("optimization/best_weights"))
    predict_cli.add_argument("--pixel-dir", type=Path, default=Path("runs/pixel/results"))
    predict_cli.add_argument("--teacher", type=Path, default=Path("runs/teacher_test.npz"))
    predict_cli.add_argument("--out", type=Path, default=Path("runs/members"))
    predict_cli.add_argument("--exact-weights", action="store_true")
    args = parser.parse_args()
    if args.command == "teacher":
        teacher_maps(args)
    else:
        {"export": export, "score-oof": score_oof, "predict": predict_members}[args.command](args, frozen)


if __name__ == "__main__":
    main()

