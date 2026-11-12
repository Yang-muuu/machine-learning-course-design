"""Recalculate the archived Task 4 development results with the standard library."""

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
IMAGE_IDS = tuple(str(index) for index in range(1, 21))


def read_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def verify_splits(splits):
    expected = set(IMAGE_IDS)
    held_out = []
    for split in splits:
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]
        if (len(train_ids) != 16 or len(val_ids) != 4 or
                set(train_ids) & set(val_ids) or set(train_ids + val_ids) != expected):
            raise ValueError("Expected a disjoint 16/4 split of the 20 images")
        held_out.extend(val_ids)
    if len(splits) != 5 or len(held_out) != len(expected) or set(held_out) != expected:
        raise ValueError("The five validation folds must hold out every image once")


def verify_weights(snapshot):
    members = snapshot["members"]
    recipe = snapshot["recipe"]
    weights = dict.fromkeys(members, 0.0)
    if recipe["initial_source"] not in members:
        raise ValueError("Blend recipe starts from an unlisted member")
    weights[recipe["initial_source"]] = 1.0
    for step in recipe["steps"]:
        alpha = step["alpha"]
        if not 0 <= alpha <= 1 or step["source"] not in members:
            raise ValueError(f"Invalid convex blend step: {step}")
        for alias in weights:
            weights[alias] *= 1 - alpha
        weights[step["source"]] += alpha
    if not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Blend weights do not sum to one")
    for alias, value in weights.items():
        if not math.isclose(value, members[alias]["weight"], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Blend weight differs for {alias}")
    return {alias: round(value, 8) for alias, value in weights.items()}


def verify_image_metrics(snapshot):
    record = read_json("records/oof_counts.json")
    if (record["evaluation_protocol"] != snapshot["evaluation_protocol"] or
            record["threshold"] != snapshot["threshold"] or
            record["postprocess"] != snapshot["postprocess"] or record["shape"] != [512, 512]):
        raise ValueError("Pixel counts and archived predictions use different settings")
    counts = record["per_image"]
    saved_dice = snapshot["per_image_dice"]
    if set(counts) != set(IMAGE_IDS) or set(saved_dice) != set(IMAGE_IDS):
        raise ValueError("Pixel counts and saved Dice must cover IDs 1 to 20")
    dice, precision, recall, iou = {}, {}, {}, {}
    for image_id in IMAGE_IDS:
        tp, fp, fn = (counts[image_id][key] for key in ("tp", "fp", "fn"))
        if (any(type(count) is not int or count < 0 for count in (tp, fp, fn)) or
                tp + fp + fn > 512 ** 2 or min(tp + fp, tp + fn) == 0):
            raise ValueError(f"Invalid pixel counts for image {image_id}")
        dice[image_id] = 2 * tp / (2 * tp + fp + fn)
        precision[image_id] = tp / (tp + fp)
        recall[image_id] = tp / (tp + fn)
        iou[image_id] = tp / (tp + fp + fn)
        if not math.isclose(dice[image_id], saved_dice[image_id], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Pixel counts disagree with saved Dice for image {image_id}")
    macro_dice = mean(dice.values())
    if not math.isclose(macro_dice, snapshot["macro_oof_dice"], abs_tol=1e-12, rel_tol=0):
        raise ValueError("Recalculated macro Dice differs from the frozen snapshot")
    return {
        "images": len(dice),
        "macro_dice": macro_dice,
        "dice_sample_std": stdev(dice.values()),
        "macro_precision": mean(precision.values()),
        "macro_recall": mean(recall.values()),
        "macro_iou": mean(iou.values()),
        "per_image_dice": dice,
    }


def verify_training_logs(snapshot):
    verified = {}
    for name in ("fr96", "convnext", "qtseg", "retfound"):
        epochs = read_json(f"configs/{name}.json")["epochs"]
        expected_best = snapshot["members"][name]["best_epochs"]
        if len(expected_best) != 5:
            raise ValueError(f"Expected five selected epochs for {name}")
        with (ROOT / f"records/training/{name}.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        expected = [(fold, epoch) for fold in range(5) for epoch in range(1, epochs + 1)]
        if [(int(row["fold"]), int(row["epoch"])) for row in rows] != expected:
            raise ValueError(f"Missing or out-of-order epoch in {name} log")
        for row in rows:
            if any(not math.isfinite(float(row[key])) or float(row[key]) < 0 for key in ("loss", "seconds")):
                raise ValueError(f"Invalid training measurement in {name} log")
        for fold, best in enumerate(expected_best):
            validation = [
                (int(row["epoch"]), float(row["legacy_val_dice_0_5"]))
                for row in rows if int(row["fold"]) == fold and row["legacy_val_dice_0_5"] != ""
            ]
            if (not validation or any(not 0 <= value <= 1 for _, value in validation) or
                    max(validation, key=lambda pair: pair[1])[0] != best):
                raise ValueError(f"Selected epoch differs from the logged validation peak: {name} fold {fold}")
        verified[name] = {"epoch_rows": len(rows), "selected_epochs": expected_best}
    return verified


def main():
    snapshot = read_json("records/final_snapshot.json")
    verify_splits(snapshot["splits"])
    result = {
        "status": "archive_checks_passed",
        "scope": "recalculation from held-out training-image counts and archived epoch logs",
        "fixed_folds": len(snapshot["splits"]),
        "validation": verify_image_metrics(snapshot),
        "blend_weights": verify_weights(snapshot),
        "training_logs": verify_training_logs(snapshot),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
