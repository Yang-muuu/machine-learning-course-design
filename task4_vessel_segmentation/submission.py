"""Prediction, lossless PNG export and comparison with the supplied converter."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from retinal import index_images, rle_decode, rle_encode, template_ids, write_json


def write_submission(path, ids, masks, foreground="black"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Predicted"])
        for key, mask in zip(ids, masks):
            writer.writerow([key, rle_encode(mask > 0 if foreground == "white" else mask == 0)])


def verify_submission(root, submission, mask_dir, foreground="black", compare_official=False):
    root, submission, mask_dir = Path(root), Path(submission), Path(mask_dir)
    expected = template_ids(root)
    with submission.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["Id", "Predicted"]:
            raise ValueError("Submission columns must be exactly Id,Predicted")
        rows = list(reader)
    actual = [row["Id"] for row in rows]
    if actual != expected:
        raise ValueError("Submission IDs/order differ from sample_submission.csv")
    paths = index_images(mask_dir)
    if set(paths) != set(expected):
        raise ValueError("Exported PNG IDs differ from template")
    fractions = []
    ordered_paths = []
    for row in rows:
        path = paths[row["Id"]]
        with Image.open(path) as im:
            if im.mode != "L" or im.size != (512, 512) or im.format != "PNG":
                raise ValueError(f"Expected lossless L-mode 512x512 PNG: {path}")
            mask = np.asarray(im)
        if not np.isin(mask, [0, 255]).all():
            raise ValueError(f"Nonbinary exported mask: {path}")
        encoded_mask = mask > 0 if foreground == "white" else mask == 0
        if not np.array_equal(rle_decode(row["Predicted"]), encoded_mask):
            raise ValueError(f"Decoded RLE differs from PNG for ID {row['Id']}")
        if rle_encode(encoded_mask) != row["Predicted"]:
            raise ValueError(f"Noncanonical RLE for ID {row['Id']}")
        fractions.append(float((mask == 0).mean()))
        ordered_paths.append(str(path))
    official_status = "not requested"
    if compare_official:
        if foreground != "white":
            raise ValueError("Official converter encodes white pixels; compare only the official-compatible file")
        script = root / "segmentation_to_csv.py"
        # This user-provided program was read and inspected before execution.
        spec = importlib.util.spec_from_file_location("supplied_segmentation_converter", script)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        encoded = official.turn_to_str(ordered_paths)
        if encoded != [row["Predicted"] for row in rows]:
            raise AssertionError("Custom encoder differs from the provided converter")
        official_status = "all 20 rows exactly match supplied turn_to_str"
    return {"status": "passed", "submission": str(submission), "rows": len(rows),
            "template_order": "exact match", "png_shape": [512, 512],
            "png_values": [0, 255], "png_vessel_value": 0,
            "rle_foreground": foreground, "rle_order": "C row-major, 1-based",
            "roundtrip": "all rows match exported PNGs", "official_comparison": official_status,
            "per_image_vessel_fraction": dict(zip(expected, fractions))}


def verify(args):
    report = verify_submission(args.data, args.submission, args.masks,
                               args.foreground, args.foreground == "white" and not args.skip_official)
    write_json(Path(args.out) / ("submission_check.json" if args.foreground == "white" else "vessel_foreground_check.json"), report)
    print(json.dumps(report, indent=2), flush=True)


def preview(root, ids, masks, destination):
    images = index_images(Path(root) / "test/image")
    selected = list(zip(ids, masks))[:4]
    canvas = Image.new("RGB", (960, 352 * len(selected)), "#f1f1f1")
    draw = ImageDraw.Draw(canvas)
    for row, (key, mask) in enumerate(selected):
        original = Image.open(images[key]).convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
        rendered = Image.fromarray(mask).resize((320, 320), Image.Resampling.NEAREST)
        vessel = np.asarray(rendered) == 0
        overlay = np.asarray(original).copy()
        overlay[vessel] = (0.4 * overlay[vessel] + 0.6 * np.array([45, 255, 85])).astype(np.uint8)
        for col, panel in enumerate([original, rendered.convert("RGB"), Image.fromarray(overlay)]):
            canvas.paste(panel, (320 * col, 352 * row + 25))
        draw.text((6, 352 * row + 6), f"Test ID {key}: image | prediction (black=vessel) | overlay", fill="black")
    canvas.save(destination)


def predict(args):
    import torch
    from vessel_model import environment, load_model, predict_one, preprocess, setup

    device = setup(args.seed, args.device)
    out = Path(args.out)
    mask_dir = out / "image"
    mask_dir.mkdir(parents=True, exist_ok=True)
    ids = template_ids(args.data)
    paths = index_images(Path(args.data) / "test/image")
    if set(paths) != set(ids):
        raise ValueError("Test IDs differ from template")
    if args.threshold is None:
        if not args.calibration:
            raise ValueError("Provide --calibration validation.json or an explicit --threshold")
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        threshold = float(calibration["threshold"])
    else:
        threshold = args.threshold
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    models, metadata = [], []
    for path in args.checkpoints:
        model, checkpoint = load_model(path, device)
        models.append(model)
        metadata.append({"path": str(path), "epoch": checkpoint["epoch"],
                         "train_ids": checkpoint["split"]["train_ids"], "val_ids": checkpoint["split"]["val_ids"]})
    masks, probabilities = [], []
    for key in ids:
        image, fov = preprocess(paths[key])
        p = np.mean([predict_one(model, image, device, not args.no_tta) for model in models], axis=0) * fov
        if not np.isfinite(p).all():
            raise FloatingPointError(f"Nonfinite test probabilities for ID {key}")
        vessel = p >= threshold
        mask = np.where(vessel, 0, 255).astype(np.uint8)
        Image.fromarray(mask).save(mask_dir / f"{key}.png")
        masks.append(mask)
        probabilities.append(p.astype(np.float32))
        print(f"Predicted ID {key}: vessel_fraction={vessel.mean():.5f}", flush=True)
    np.savez_compressed(out / "test_probabilities.npz", ids=np.array(ids), probabilities=np.stack(probabilities))
    write_submission(out / "submission_background_legacy_do_not_submit.csv", ids, masks, "white")
    write_submission(out / "submission.csv", ids, masks, "black")
    write_submission(out / "submission_vessel_foreground.csv", ids, masks, "black")
    preview(args.data, ids, masks, out / "prediction_preview.png")
    report = {"models": metadata, "environment": environment(device), "threshold": threshold,
              "calibration_file": args.calibration, "tta": not args.no_tta,
              "png": "black vessel=0, white background=255, lossless 512x512 PNG",
              "submission.csv": "Confirmed by user leaderboard feedback: encode black vessel pixels.",
              "submission_vessel_foreground.csv": "Alias of the confirmed vessel-foreground submission.",
              "submission_background_legacy_do_not_submit.csv": "Historical background encoding; user confirmed near-zero score. Do not submit.",
              "rle_foreground_confirmed": "vessels; user feedback 2026-11-01"}
    write_json(out / "prediction_config.json", report)
    check = verify_submission(args.data, out / "submission.csv", mask_dir)
    write_json(out / "submission_check.json", check)
    alternate = verify_submission(args.data, out / "submission_vessel_foreground.csv", mask_dir, "black", False)
    write_json(out / "vessel_foreground_check.json", alternate)
    print(f"Saved and verified {out / 'submission.csv'}; threshold={threshold:.3f}", flush=True)

    # Compact checkpoints retain model construction and split provenance, without
    # optimizer state. They are sufficient for offline prediction and packaging.
    for i, (path, model, meta) in enumerate(zip(args.checkpoints, models, metadata)):
        torch.save({"model": {key: value.cpu() for key, value in model.state_dict().items()},
                    "config": {"width": model.width}, "epoch": meta["epoch"],
                    "split": {"train_ids": meta["train_ids"], "val_ids": meta["val_ids"]}}, out / f"inference_model_{i}.pt")
