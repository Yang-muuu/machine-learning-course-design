"""Rebuild the submitted four-model CSV from the frozen prediction archive."""

import argparse
import csv
import hashlib
import io
import json
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "records/kaggle_final_recipe.json"


def read_member(archive, name, digest):
    member = archive.getinfo(name)
    if member.file_size > 2_000_000:
        raise ValueError(f"Unexpected member size: {name}")
    data = archive.read(member)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Archived input checksum mismatch: {name}")
    return data


def build(archive_path, recipe):
    models = recipe["models"]
    weights = [Decimal(model["weight"]) for model in models]
    if len(models) != 4 or any(not w.is_finite() or w <= 0 for w in weights) or sum(weights) != 1:
        raise ValueError("Expected four positive weights with sum one.")
    with ZipFile(archive_path) as archive:
        template = read_member(archive, recipe["template"], recipe["template_sha256"])
        reader = csv.DictReader(io.StringIO(template.decode("utf-8-sig"), newline=""))
        if reader.fieldnames != ["ID", "Emotion"]:
            raise ValueError("Expected the recorded ID,Emotion template.")
        ids = [row["ID"] for row in reader]
        if len(ids) != recipe["rows"] or len(set(ids)) != len(ids) or not all(ids):
            raise ValueError("Template must contain 7,065 distinct image IDs.")

        bank = []
        for model in models:
            raw = read_member(archive, model["prediction_member"], model["prediction_sha256"])
            probabilities = np.load(io.BytesIO(raw), allow_pickle=False)
            expected_shape = (len(ids), len(recipe["classes"]))
            if (probabilities.shape != expected_shape
                or not np.issubdtype(probabilities.dtype, np.floating)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
                raise ValueError(f"Invalid probability table: {model['key']}")
            bank.append(probabilities)

    classes = sum(float(w) * p for w, p in zip(weights, bank)).argmax(axis=1)
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(["ID", "Emotion"])
    writer.writerows(zip(ids, map(int, classes)))
    payload = out.getvalue().encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != recipe["submission_sha256"]:
        raise ValueError("Rebuilt file does not match the verified Kaggle CSV.")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="original frozen_predictions.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "work/kaggle_final_submission.csv")
    args = parser.parse_args()
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    payload = build(args.archive, recipe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise FileExistsError(args.output)
    args.output.write_bytes(payload)
    print(f"{recipe['rows']:,} rows · SHA-256 {recipe['submission_sha256']} · {args.output}")


if __name__ == "__main__":
    main()
