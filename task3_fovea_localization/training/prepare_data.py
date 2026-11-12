"""Audit the supplied competition files without altering the originals."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import StratifiedGroupKFold


def contact_sheet(rows, root, split, output, ncols=5, thumb=(360, 260)):
    nrows = (len(rows) + ncols - 1) // ncols
    canvas = Image.new("RGB", (ncols * thumb[0], nrows * (thumb[1] + 30)), "#161a22")
    for i, row in enumerate(rows):
        im = Image.open(root / split / row["filename"]).convert("RGB")
        scale = min(thumb[0] / im.width, thumb[1] / im.height)
        im = im.resize((round(im.width * scale), round(im.height * scale)))
        draw = ImageDraw.Draw(im)
        if "x" in row:
            x, y = row["x"] * scale, row["y"] * scale
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="#00ffb7", width=2)
            if "xmin" in row:
                draw.rectangle(tuple(row[k] * scale for k in ("xmin", "ymin", "xmax", "ymax")), outline="#ffb84d", width=1)
        ox, oy = (i % ncols) * thumb[0], (i // ncols) * (thumb[1] + 30)
        canvas.paste(im, (ox, oy))
        ImageDraw.Draw(canvas).text((ox + 8, oy + thumb[1] + 5), row["filename"], fill="white")
    canvas.save(output)


def prepare(root: Path, output: Path, seed=2026):
    output.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(root / "fovea_localization_train_GT.csv")
    assert list(gt.columns) == ["data", "Fovea_X", "Fovea_Y"]
    assert gt["data"].is_unique
    rows, duplicates = [], defaultdict(list)
    for row in gt.itertuples(index=False):
        filename = f"{int(row.data):04d}.jpg"
        path = root / "train" / filename
        with Image.open(path) as im:
            width, height = im.size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        duplicates[digest].append(filename)
        visible = (row.Fovea_X != 0) or (row.Fovea_Y != 0)
        assert not visible or (0 <= row.Fovea_X < width and 0 <= row.Fovea_Y < height)
        rec = dict(image_id=int(row.data), filename=filename, width=width, height=height,
                   x=float(row.Fovea_X), y=float(row.Fovea_Y), visible=int(visible),
                   group=digest, stratum=f"{width}x{height}")
        tree = ET.parse(root / "train_location" / f"{int(row.data):04d}.xml").getroot()
        assert int(tree.findtext("size/width")) == width
        assert int(tree.findtext("size/height")) == height
        box = tree.find("object/bndbox")
        if box is not None:
            rec.update({k: int(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")})
            assert rec["xmin"] <= rec["x"] <= rec["xmax"]
            assert rec["ymin"] <= rec["y"] <= rec["ymax"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["fold"] = -1
    # Images with identical bytes always stay in the same fold.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, va) in enumerate(splitter.split(df, df.stratum, df.group)):
        df.loc[va, "fold"] = fold
    assert df.groupby("group").fold.nunique().max() == 1
    df.to_csv(output / "train.csv", index=False)
    test_rows = []
    for path in sorted((root / "test").glob("*.jpg")):
        with Image.open(path) as im:
            width, height = im.size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest not in duplicates, "Train/test duplicate found: review before training"
        test_rows.append(dict(image_id=int(path.stem), filename=path.name, width=width, height=height))
    pd.DataFrame(test_rows).to_csv(output / "test.csv", index=False)
    sample = pd.read_csv(root / "sample_submission.csv")
    expected = [f"{r['image_id']}_Fovea_{axis}" for r in test_rows for axis in ("X", "Y")]
    assert sample.ImageID.tolist() == expected
    audit = dict(train_count=len(df), test_count=len(test_rows), submission_rows=len(sample),
                 visible_train=int(df.visible.sum()), invisible_train=int((df.visible == 0).sum()),
                 dimensions=df.groupby(["width", "height"]).size().to_dict(),
                 exact_duplicate_groups=[v for v in duplicates.values() if len(v) > 1],
                 fold_counts=df.fold.value_counts().sort_index().to_dict(),
                 split_seed=seed, split_method="StratifiedGroupKFold, image dimensions strata, SHA256 groups",
                 test_labels_used=False)
    audit["dimensions"] = {f"{w}x{h}": int(n) for (w, h), n in audit["dimensions"].items()}
    (output / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(rows, root, "train", output / "train_contact_sheet.jpg")
    contact_sheet(test_rows, root, "test", output / "test_contact_sheet.jpg")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/detection"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    prepare(args.data, args.output)
