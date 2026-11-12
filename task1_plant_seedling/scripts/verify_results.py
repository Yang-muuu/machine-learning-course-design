"""Recompute report metrics and optionally check a submission CSV offline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def read_table(path, header):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(header):
            raise ValueError(f"{path.name}: expected columns {list(header)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name}: incomplete or extra fields")
    return rows


def expect_close(actual, expected, name, tolerance=1e-10):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{name}: {actual} != {expected}")


def verify_oof(summary):
    classes = summary["classes"]
    report = summary["supervised"]
    rows = read_table(RESULTS / "oof_predictions.csv", ("sample_index", "true_class", "pred_class", "confidence"))
    if len(rows) != 500 or sorted(int(r["sample_index"]) for r in rows) != list(range(500)):
        raise ValueError("OOF sample indices must cover exactly 0–499")
    if Counter(r["true_class"] for r in rows) != Counter(summary["train_counts"]):
        raise ValueError("OOF class support does not match the training set")
    lut = {c: k for k, c in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for row in rows:
        conf = float(row["confidence"])
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("Invalid OOF confidence")
        matrix[lut[row["true_class"]]][lut[row["pred_class"]]] += 1
    if matrix != report["confusion_rows_true_columns_pred"]:
        raise ValueError("OOF confusion matrix mismatch")
    correct = sum(matrix[k][k] for k in range(len(classes)))
    expect_close(correct, report["oof_correct"], "correct predictions")
    expect_close(500 - correct, report["oof_errors"], "prediction errors")
    f1 = [2 * matrix[k][k] / (sum(matrix[k]) + sum(row[k] for row in matrix)) for k in range(len(classes))]
    for k, c in enumerate(classes):
        expect_close(f1[k], report["per_class_f1"][k], c)
    expect_close(correct / 500, report["oof_accuracy"], "accuracy")
    expect_close(statistics.mean(f1), report["oof_macro_f1"], "macro-F1")
    return {"n": 500, "correct": correct, "accuracy": correct / 500,
            "macro_f1": statistics.mean(f1), "per_class_f1": dict(zip(classes, f1)),
            "confusion_rows_true_columns_pred": matrix}


def verify_logs(summary):
    epoch_rows = read_table(RESULTS / "epochs.csv",
                            ("fold", "epoch", "total_epochs", "train_loss", "val_accuracy", "val_macro_f1",
                             "epoch_seconds", "best_weight"))
    expected = [(k, i) for k in range(1, 6) for i in range(1, 41)]
    if [(int(r["fold"]), int(r["epoch"])) for r in epoch_rows] != expected:
        raise ValueError("Expected five complete 40-epoch folds")
    for row in epoch_rows:
        loss, accuracy, f1, seconds = (float(row[k]) for k in ("train_loss", "val_accuracy", "val_macro_f1", "epoch_seconds"))
        if (int(row["total_epochs"]) != 40 or row["best_weight"] not in ("", "ema", "raw")
                or not all(map(math.isfinite, (loss, accuracy, f1, seconds)))
                or loss < 0 or seconds < 0 or not 0 <= accuracy <= 1 or not 0 <= f1 <= 1):
            raise ValueError("Invalid supervised training record")
    for entry in summary["supervised"]["fold_results"]:
        rows = [r for r in epoch_rows if int(r["fold"]) == entry["fold"]]
        updates = [r for r in rows if r["best_weight"]]
        if not updates:
            raise ValueError("Fold has no saved checkpoint")
        best = updates[-1]
        if int(best["epoch"]) != entry["best_epoch"] or best["best_weight"] != entry["selected_weight"]:
            raise ValueError("Best checkpoint mismatch")
        expect_close(float(best["val_macro_f1"]), entry["val_macro_f1"], "logged best F1", 0.00005)
        expect_close(float(best["val_macro_f1"]), max(float(row["val_macro_f1"]) for row in rows),
                     "fold maximum logged F1")
    students = read_table(RESULTS / "student_epochs.csv",
                          ("fold", "epoch", "loss", "mixed_batches", "amp_overflow_batches", "successful_updates"))
    if [(int(r["fold"]), int(r["epoch"])) for r in students] != [(k, i) for k in range(1, 6) for i in range(1, 6)]:
        raise ValueError("Expected five complete five-epoch student histories")
    if any(not math.isfinite(float(r["loss"])) or int(r["successful_updates"]) <= 0 for r in students):
        raise ValueError("Invalid student training record")
    return {"supervised_epochs": len(epoch_rows), "student_epochs": len(students)}


def submission_rows(path, classes, validate_classes=True):
    rows = read_table(path, ("ID", "Category"))
    ids = [row["ID"] for row in rows]
    if len(rows) != 378 or len(set(ids)) != 378 or any(not x.strip() for x in ids):
        raise ValueError(f"{path.name}: expected 378 nonempty unique IDs")
    if validate_classes and any(r["Category"] not in classes for r in rows):
        raise ValueError(f"{path.name}: unexpected category")
    return rows


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_submission(args, summary):
    classes = summary["classes"]
    rows = submission_rows(args.submission, classes)
    items = [(r["ID"], r["Category"]) for r in rows]
    current_hash = sha(args.submission.read_bytes())
    key_hash = sha(json.dumps(sorted([list(x) for x in items]), ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8"))
    expected = summary["final_submission"]
    result = {"rows": len(rows), "category_counts": dict(Counter(r["Category"] for r in rows)),
              "sha256": current_hash,
              "matches_historical_file_bytes": current_hash == expected["sha256"],
              "matches_historical_id_category_mapping": key_hash == expected["sorted_id_category_sha256"]}
    if args.template:
        template = submission_rows(args.template, classes, validate_classes=False)
        if [r["ID"] for r in rows] != [r["ID"] for r in template]:
            raise ValueError("CSV IDs or row order differ from the template")
        result["template_order_match"] = True
    if args.compare:
        other = {r["ID"]: r["Category"] for r in submission_rows(args.compare, classes)}
        if set(other) != {i for i, _ in items}:
            raise ValueError("Comparison CSV has different IDs")
        result["changed_predictions_vs_compare"] = sum(other[i] != c for i, c in items)
    if args.expect_historical_mapping and not result["matches_historical_id_category_mapping"]:
        raise ValueError("ID/category mapping does not match the archived submission")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, help="Optional result CSV to check")
    parser.add_argument("--template", type=Path, help="Optional original submission template")
    parser.add_argument("--compare", type=Path, help="Optional second prediction CSV to compare by ID")
    parser.add_argument("--expect-historical-mapping", action="store_true",
                        help="Require the archived ID/category mapping; CSV byte formatting may differ")
    args = parser.parse_args()
    if (args.template or args.compare or args.expect_historical_mapping) and not args.submission:
        parser.error("--template, --compare and --expect-historical-mapping require --submission")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    try:
        result = {"scope": "archived_training_predictions_and_epoch_logs",
                  "oof": verify_oof(summary), "logged_epochs": verify_logs(summary)}
        if args.submission:
            result["submission_check"] = verify_submission(args, summary)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
