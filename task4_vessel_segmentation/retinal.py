"""Reproducible 2026task-4 vessel segmentation and official-compatible RLE.

Training labels: white=vessel. Export PNGs: black=vessel, white=background.
The supplied converter encodes nonzero pixels; competition CSVs encode vessels,
as confirmed by the user's leaderboard feedback. Export foreground explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def image_paths(directory):
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    paths = [p for p in Path(directory).iterdir() if p.suffix.lower() in extensions]
    return sorted(paths, key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))


def template_ids(root):
    with (Path(root) / "sample_submission.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["Id", "Predicted"]:
            raise ValueError("Template columns must be Id,Predicted")
        ids = [row["Id"] for row in reader]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicated template IDs")
    return ids


def index_images(directory):
    paths = image_paths(directory)
    result = {p.stem: p for p in paths}
    if len(result) != len(paths):
        raise ValueError(f"Duplicate image stems in {directory}")
    return result


def rle_encode(binary):
    """Encode true pixels in row-major order; start indices are 1-based."""
    flat = np.asarray(binary, dtype=bool).ravel(order="C").astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flat, [0]))))
    if not len(edges):
        return ""
    pairs = np.column_stack((edges[::2] + 1, edges[1::2] - edges[::2]))
    return " ".join(map(str, pairs.ravel()))


def rle_decode(encoded, shape=(512, 512)):
    """Validate and decode RLE, rejecting overlaps, negatives and out-of-range runs."""
    items = encoded.strip().split()
    if len(items) % 2:
        raise ValueError("RLE requires start,length pairs")
    try:
        numbers = [int(v) for v in items]
    except ValueError as exc:
        raise ValueError("Noninteger RLE token") from exc
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    previous_end = 0
    for start, length in zip(numbers[::2], numbers[1::2]):
        if start < 1 or length < 1 or start <= previous_end or start + length - 1 > flat.size:
            raise ValueError(f"Illegal RLE pair: {start}, {length}")
        flat[start - 1:start - 1 + length] = True
        previous_end = start + length - 1
    return flat.reshape(shape)


def check_rle():
    # Non-square and edge-touching examples catch transpose/off-by-one mistakes.
    examples = [np.zeros((3, 5), bool), np.ones((3, 5), bool),
                np.array([[1, 1, 0, 0, 1], [0, 1, 0, 1, 1]], bool)]
    rng = np.random.default_rng(42)
    examples.append(rng.random((17, 19)) > 0.6)
    for mask in examples:
        assert np.array_equal(rle_decode(rle_encode(mask), mask.shape), mask)
    assert rle_encode(np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]], bool)) == "1 3 10 2"
    for bad in ["0 1", "1 -2", "1 3 2 1", "15 2", "1", "a 2"]:
        try:
            rle_decode(bad, (3, 5))
        except ValueError:
            continue
        raise AssertionError(f"Accepted illegal RLE: {bad}")


def inspect(args):
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, labels, test = [index_images(root / p) for p in
                           ("train/image", "train/label", "test/image")]
    ids = template_ids(root)
    if set(train) != set(labels) or set(test) != set(ids):
        raise ValueError("Image-label-template IDs do not match")
    if len(train) != 20 or len(test) != 20:
        raise ValueError("Expected 20 training and 20 test images")
    records = []
    for split, images in [("train", train), ("test", test)]:
        for key, path in images.items():
            with Image.open(path) as im:
                size = im.size
                record = {"split": split, "id": key, "size_wh": list(size), "mode": im.mode}
            if split == "train":
                with Image.open(labels[key]) as im:
                    if im.size != size:
                        raise ValueError(f"Unaligned image and mask for {key}")
                    y = np.asarray(im.convert("L"))
                record.update(label_values=np.unique(y).tolist(), vessel_fraction=float((y >= 128).mean()))
            records.append(record)
    # Overlay genuine labels for human confirmation of vessel polarity.
    selected = list(train)[:4]
    canvas = Image.new("RGB", (3 * 320, len(selected) * 352), "#f3f3f3")
    draw = ImageDraw.Draw(canvas)
    for row, key in enumerate(selected):
        rgb = Image.open(train[key]).convert("RGB").resize((320, 320))
        y = Image.open(labels[key]).convert("L").resize((320, 320), Image.Resampling.NEAREST)
        a = np.asarray(rgb).copy()
        vessel = np.asarray(y) >= 128
        a[vessel] = (0.35 * a[vessel] + 0.65 * np.array([40, 255, 80])).astype(np.uint8)
        for col, panel in enumerate([rgb, y.convert("RGB"), Image.fromarray(a)]):
            canvas.paste(panel, (col * 320, row * 352 + 25))
        draw.text((6, row * 352 + 6), f"ID {key}: image | provided label (white=vessel) | overlay", fill="black")
    canvas.save(out / "data_preview.png")
    check_rle()
    report = {"train_count": len(train), "test_count": len(test), "output_size_wh": [512, 512],
              "label_vessel_polarity": "white>=128 (confirm in data_preview.png)",
              "export_vessel_value": 0, "export_background_value": 255,
              "official_rle_encodes": "nonzero pixels = white background in requested PNGs",
              "records": records, "rle_roundtrip": "passed"}
    write_json(out / "data_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/segmentation")
    parser.add_argument("--out", default="outputs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    for name in ("check", "train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--device", default="auto")
        command.add_argument("--seed", type=int, default=42)
        if name in ("check", "train"):
            command.add_argument("--width", type=int, default=24)
            command.add_argument("--patch", type=int, default=160)
            command.add_argument("--batch", type=int, default=8)
        if name == "train":
            command.add_argument("--fold", default="0", choices=["0", "1", "2", "3", "4", "all"])
            command.add_argument("--epochs", type=int, default=100)
            command.add_argument("--steps", type=int, default=40)
            command.add_argument("--lr", type=float, default=0.001)
            command.add_argument("--validate-every", type=int, default=2)
            command.add_argument("--patience", type=int, default=15,
                                 help="Number of validation checks without improvement; 0 disables")
        elif name == "evaluate":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--no-tta", action="store_true")
    command = sub.add_parser("predict")
    command.add_argument("--checkpoints", nargs="+", required=True)
    command.add_argument("--calibration")
    command.add_argument("--threshold", type=float)
    command.add_argument("--device", default="auto")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--no-tta", action="store_true")
    command = sub.add_parser("verify")
    command.add_argument("--submission", default="outputs/submission.csv")
    command.add_argument("--masks", default="outputs/image")
    command.add_argument("--foreground", choices=["white", "black"], default="black")
    command.add_argument("--skip-official", action="store_true")
    command = sub.add_parser("calibrate")
    command.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"], choices=["0", "1", "2", "3", "4"])
    command.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "inspect":
        inspect(args)
    elif args.command in ("predict", "verify"):
        import submission
        getattr(submission, args.command)(args)
    else:
        import vessel_model
        getattr(vessel_model, args.command)(args)


if __name__ == "__main__":
    main()
