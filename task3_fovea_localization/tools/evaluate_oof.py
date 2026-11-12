"""Recompute the grouped OOF results from the official training coordinates."""
import argparse
import csv
import hashlib
import json
from math import dist, fsum, isfinite, sqrt
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(truth, points):
    count = len(truth)
    sx, sy = (fsum((points[i][axis] - truth[i][axis]) ** 2 for i in truth)
              for axis in range(2))
    distances = {i: dist(truth[i], points[i]) for i in truth}
    worst = max(distances, key=distances.get)
    return {
        "x_rmse": sqrt(sx / count),
        "y_rmse": sqrt(sy / count),
        "oof_rmse": sqrt((sx + sy) / (2 * count)),
        "median_euclidean_px": median(distances.values()),
        "worst_image_id": worst,
        "worst_euclidean_px": distances[worst],
        "worst_squared_error_share": distances[worst] ** 2 / (sx + sy) if sx + sy else 0.0,
    }


def evaluate(labels):
    fold_rows = rows(ROOT / "configs/fold_assignments.csv")
    folds = {int(row["image_id"]): int(row["fold"]) for row in fold_rows}
    label_rows = rows(labels)
    truth = {int(Path(row["data"]).stem): (float(row["Fovea_X"]), float(row["Fovea_Y"]))
             for row in label_rows}
    if len(fold_rows) != 80 or len(label_rows) != 80 or set(truth) != set(range(1, 81)) or set(folds) != set(truth):
        raise ValueError("Expected each of the 80 recorded training image IDs exactly once.")
    if any(not all(isfinite(value) for value in point) for point in truth.values()):
        raise ValueError("Nonfinite training coordinates.")
    groups = {}
    for row in fold_rows:
        fold = int(row["fold"])
        if groups.setdefault(row["group"], fold) != fold:
            raise ValueError("Identical image bytes cross validation folds.")
    branch_points = {}
    for file in sorted((ROOT / "predictions/oof").glob("*.csv")):
        data = rows(file)
        if len(data) != len(folds) or {int(r["image_id"]): int(r["fold"]) for r in data} != folds:
            raise ValueError(f"Unexpected fold membership in {file.name}")
        points = {int(r["image_id"]): (float(r["identity_x"]), float(r["identity_y"])) for r in data}
        if any(not all(isfinite(value) for value in point) for point in points.values()):
            raise ValueError(f"Nonfinite predictions in {file.name}")
        branch_points[file.stem] = points
    plan = json.loads((ROOT / "configs/fusion.json").read_text(encoding="utf-8"))
    weights = [{name: float(weight) for name, weight in plan["weights"][axis].items()}
               for axis in ("X", "Y")]
    for axis in weights:
        if fsum(axis.values()) != 1.0 or not axis.keys() <= branch_points.keys():
            raise ValueError("Invalid OOF blend weights.")
    final = {
        i: tuple(fsum(weight * branch_points[name][i][k] for name, weight in weights[k].items())
                 for k in range(2))
        for i in truth
    }
    return {
        "training_rows": len(truth),
        "image_hash_groups": len(groups),
        "fold_sizes": [list(folds.values()).count(i) for i in range(5)],
        "training_label_file_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
        "branches": {name: summarize(truth, points) for name, points in branch_points.items()},
        "equal_fusion": summarize(truth, final),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/detection/fovea_localization_train_GT.csv")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.labels), indent=2))


if __name__ == "__main__":
    main()
