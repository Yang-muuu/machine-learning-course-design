"""Recompute the three recorded validation folds using only the standard library."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / "records"
CLASSES = ("angry", "fear", "happy", "sad", "surprise", "neutral")
MODELS = ("ResNet18", "ResNet34")
METHODS = (*MODELS, "Equal_fusion")
FOLDS = (0, 1, 2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def distribution(values):
    probabilities = tuple(map(float, values))
    total = math.fsum(probabilities)
    require(all(0 <= value <= 1 for value in probabilities), "Invalid probability")
    require(abs(total - 1) <= 1e-5, "Probabilities must sum to one")
    return tuple(value / total for value in probabilities)


def three_predictions(first, second):
    return first, second, tuple((a + b) / 2 for a, b in zip(first, second, strict=True))


def measures(rows, model_index):
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    losses = []
    for label, *predictions in rows:
        probability = predictions[model_index]
        prediction = max(range(len(CLASSES)), key=probability.__getitem__)
        confusion[label][prediction] += 1
        losses.append(-math.log(max(probability[label], 1e-15)))
    actual = [sum(row) for row in confusion]
    predicted = [sum(column) for column in zip(*confusion)]
    diagonal = [confusion[index][index] for index in range(len(CLASSES))]
    recall = [tp / n if n else 0.0 for tp, n in zip(diagonal, actual)]
    f1 = [2 * tp / (a + p) if a + p else 0.0 for tp, a, p in zip(diagonal, actual, predicted)]
    return {
        "correct_count": sum(diagonal),
        "validation_count": len(rows),
        "accuracy": sum(diagonal) / len(rows),
        "macro_f1": math.fsum(f1) / len(CLASSES),
        "balanced_accuracy": math.fsum(recall) / len(CLASSES),
        "log_loss": math.fsum(losses) / len(rows),
        **{f"{name}_recall": value for name, value in zip(CLASSES, recall)},
        "confusion_matrix": confusion,
    }


def group_ledger(audit):
    ledger = {}
    for row in load_csv(RECORDS / "split_groups.csv.gz"):
        group, label, fold, count = (int(row[key]) for key in ("group", "label", "fold", "count"))
        require(group not in ledger, "The same pixel group appears twice in the split")
        require(0 <= label < len(CLASSES) and 0 <= fold < 5 and count > 0, "Bad split group")
        ledger[group] = (label, fold, count)
    counts_by_fold = Counter()
    counts_by_label = Counter()
    for label, fold, count in ledger.values():
        counts_by_fold[fold] += count
        counts_by_label[label] += count
    require([counts_by_fold[i] for i in range(5)] == audit["fold_counts"], "Fold size mismatch")
    require(sum(counts_by_fold.values()) == audit["clean_train_count"], "Clean size mismatch")
    require(audit["clean_train_count"] + audit["quarantined_train_count"] == audit["raw_train_count"],
            "Clean and quarantined counts do not match raw train count")
    for name, label in audit["class_to_label"].items():
        require(counts_by_label[label] == audit["clean_class_counts"][name], "Class count mismatch")
    require(sum(count > 1 for _, _, count in ledger.values()) == audit["same_label_duplicate_groups"],
            "Duplicate-group count mismatch")
    require(sum(count - 1 for _, _, count in ledger.values()) == audit["same_label_duplicate_excess_images"],
            "Duplicate-image count mismatch")
    return ledger


def archived_fold(fold, ledger):
    rows = load_csv(RECORDS / f"fold_{fold}_validation.csv.gz")
    group_counts = Counter()
    result = []
    for row in rows:
        group, label = int(row["group"]), int(row["label"])
        expected_label, expected_fold, _ = ledger[group]
        require((label, fold) == (expected_label, expected_fold), "Validation label or fold mismatch")
        group_counts[group] += 1
        first, second = (distribution(row[f"{model.lower()}_{name}"] for name in CLASSES) for model in MODELS)
        result.append((label, *three_predictions(first, second)))
    expected = Counter({group: count for group, (_, f, count) in ledger.items() if f == fold})
    require(group_counts == expected, f"Incomplete validation coverage in fold {fold}")
    return result


def check_measures(computed, recorded, scope):
    for metric, value in computed.items():
        if metric != "confusion_matrix":
            require(math.isclose(value, recorded[metric], rel_tol=0, abs_tol=1e-7),
                    f"{scope}: {metric} {value} differs from recorded {recorded[metric]}")


def check_history(path, best_epoch, accuracy):
    history = load_csv(path)
    require([int(row["epoch"]) for row in history] == list(range(1, 26)), f"{path.name}: incomplete history")
    for row in history:
        require(all(math.isfinite(float(value)) for value in row.values()), f"{path.name}: invalid history")
    best = max(history, key=lambda row: (float(row["val_accuracy"]), -float(row["val_log_loss"])))
    require(int(best["epoch"]) == best_epoch, f"{path.name}: wrong best epoch")
    require(math.isclose(float(best["val_accuracy"]), accuracy, rel_tol=0, abs_tol=1e-12),
            f"{path.name}: best validation accuracy mismatch")


def split_signature(path):
    fields = ["path", "ID", "label", "pixel_hash", "validation_fold"]
    rows = load_csv(path)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in rows)
    return hashlib.sha256(output.getvalue().encode("utf-8")).hexdigest()


def verify(split_manifest=None):
    audit = json.loads((RECORDS / "data_audit.json").read_text(encoding="utf-8"))
    expected = json.loads((RECORDS / "summary.json").read_text(encoding="utf-8"))
    for filename, key in (("train_emotion.py", "trainer_sha256"), ("prepare_data.py", "prepare_data_sha256")):
        observed = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        require(observed == expected[key], f"Source SHA-256 mismatch: {filename}")
    require(tuple(expected["validation_folds"]) == FOLDS, "Expected training-side folds 0, 1 and 2")
    require(expected["fusion_weights"] == {name: 0.5 for name in MODELS}, "Unexpected fusion weights")
    if split_manifest:
        require(split_signature(split_manifest) == audit["fixed_split_sha256_lf"], "Rebuilt split signature differs")
    ledger = group_ledger(audit)
    by_fold = {fold: archived_fold(fold, ledger) for fold in FOLDS}
    per_fold = {}
    for fold, rows in by_fold.items():
        per_fold[fold] = {}
        for index, method in enumerate(METHODS):
            actual = measures(rows, index)
            recorded = expected["per_fold"][str(fold)][method]
            check_measures(actual, recorded, f"fold {fold}, {method}")
            if method in MODELS:
                check_history(RECORDS / "history" / f"{method.lower()}_fold_{fold}.csv",
                              recorded["best_epoch"], actual["accuracy"])
                actual["best_epoch"] = recorded["best_epoch"]
            per_fold[fold][method] = actual

    screening = {row["lr"]: row for row in expected["resnet34_lr_screening_fold_0"]}
    require(set(screening) == {0.0005, 0.0003}, "LR screening rows differ")
    check_history(RECORDS / "history/resnet34_lr5e4_fold_0.csv", screening[0.0005]["best_epoch"],
                  screening[0.0005]["accuracy"])
    winner = max(screening, key=lambda lr: (screening[lr]["accuracy"], screening[lr]["macro_f1"],
                                           -screening[lr]["log_loss"]))
    require(winner == 0.0003 and all(expected["per_fold"][str(f)]["ResNet34"]["lr"] == winner for f in FOLDS),
            "ResNet34 confirmation recipe differs from the selected learning rate")

    pooled_rows = [row for rows in by_fold.values() for row in rows]
    pooled = {}
    for index, method in enumerate(METHODS):
        pooled[method] = measures(pooled_rows, index)
        pooled[method]["confirmation_mean_accuracy"] = (
            per_fold[1][method]["accuracy"] + per_fold[2][method]["accuracy"]) / 2
        check_measures(pooled[method], expected["pooled"][method], f"pooled {method}")
    return {
        "scope": "archived training-set validation, folds 0-2 of a fixed five-way split",
        "fold_counts": audit["fold_counts"],
        "verified_validation_rows": len(pooled_rows),
        "verified_epoch_rows": 7 * 25,
        "split_manifest_verified": bool(split_manifest),
        "per_fold": per_fold,
        "pooled": pooled,
    }


def compare_runs(resnet18, resnet34):
    """Recompute the metrics from newly trained fold output directories."""
    by_fold = {}
    seen_ids, seen_groups = set(), set()
    for fold in FOLDS:
        rows_a, rows_b = (load_csv(root / f"fold_{fold}/validation_predictions.csv")
                          for root in (resnet18, resnet34))
        first = {row["ID"]: row for row in rows_a}
        second = {row["ID"]: row for row in rows_b}
        require(len(first) == len(rows_a) and len(second) == len(rows_b) and first.keys() == second.keys(),
                f"fold {fold}: model validation IDs differ or repeat")
        require(not seen_ids.intersection(first), "Validation ID occurs in more than one fold")
        groups = {row["pixel_hash"] for row in first.values()}
        require(not seen_groups.intersection(groups), "Pixel group crosses validation folds")
        seen_ids.update(first)
        seen_groups.update(groups)
        result = []
        for key, a in first.items():
            b = second[key]
            require(all(a[k] == b[k] for k in ("label", "pixel_hash", "path")), "Row metadata differs")
            label = int(a["label"])
            require(0 <= label < len(CLASSES), "Bad validation label")
            pair = (distribution(row[f"p_{name.capitalize()}"] for name in CLASSES) for row in (a, b))
            result.append((label, *three_predictions(*pair)))
        by_fold[fold] = result
    return {
        "scope": "recomputed training-set validation for supplied runs",
        "per_fold": {f: {m: measures(rows, i) for i, m in enumerate(METHODS)} for f, rows in by_fold.items()},
        "pooled": {m: measures([row for rows in by_fold.values() for row in rows], i)
                   for i, m in enumerate(METHODS)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, help="compare the generated split with the recorded split")
    parser.add_argument("--runs", type=Path, nargs=2, metavar=("RESNET18_DIR", "RESNET34_DIR"),
                        help="recalculate metrics from new training outputs")
    args = parser.parse_args()
    if args.runs and args.split_manifest:
        parser.error("--runs and --split-manifest are separate checks")
    result = compare_runs(*args.runs) if args.runs else verify(args.split_manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
