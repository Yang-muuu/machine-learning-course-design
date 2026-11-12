"""Offline checks for the exact records used in the course report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import isclose
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(folder, script, *args):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-I", "-S", "-B", script, *map(str, args)],
        cwd=ROOT / folder, capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"{folder}/{script} failed:\n{result.stderr}")
    return result.stdout


def verify_task3(labels):
    folder = "task3_fovea_localization"
    task = ROOT / folder
    hashes = load(task / "records/training_source_sha256.json")
    for name, expected in hashes.items():
        if digest(task / "training" / name) != expected:
            raise ValueError(f"Task 3 source checksum mismatch: {name}")

    plan = load(task / "configs/fusion.json")
    log_rows = 0
    for name, branch in plan["branches"].items():
        expected_epochs = list(range(1, branch["train_epochs"] + 1))
        for fold in range(5):
            with (task / "records" / name / f"fold{fold}_history.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            if [int(row["epoch"]) for row in rows] != expected_epochs or any(
                int(row["fold"]) != fold for row in rows
            ):
                raise ValueError(f"Incomplete Task 3 log: {name}, fold {fold}.")
            log_rows += len(rows)

    with tempfile.TemporaryDirectory() as temp:
        rebuilt = Path(temp) / "submission.csv"
        run(folder, "tools/rebuild_submission.py", "--verify", "--output", rebuilt)
        if digest(rebuilt) != digest(task / "predictions/submission.csv"):
            raise ValueError("Task 3 candidate differs from its rebuilt CSV.")
    summary = load(task / "records/oof_summary.json")
    result = {
        "fold_epoch_rows_checked": log_rows,
        "source_files_checked": len(hashes),
        "rebuilt_candidate_rows": 40,
        "candidate_sha256": plan["submission_sha256"],
        "archived_training_oof_rmse_px": summary["equal_fusion"]["oof_rmse"],
        "recomputed_with_official_training_labels": False,
    }
    if labels is not None:
        actual = json.loads(run(folder, "tools/evaluate_oof.py", "--labels", labels))
        if actual["training_label_file_sha256"] != summary["training_label_file_sha256"]:
            raise ValueError("Task 3 training label file does not match the archived fingerprint.")
        for name, metrics in summary["branches"].items():
            for metric in ("x_rmse", "y_rmse", "oof_rmse"):
                if not isclose(actual["branches"][name][metric], metrics[metric], abs_tol=1e-8):
                    raise ValueError(f"Task 3 archived OOF mismatch: {name}/{metric}.")
        if not isclose(actual["equal_fusion"]["oof_rmse"], result["archived_training_oof_rmse_px"], abs_tol=1e-8):
            raise ValueError("Task 3 blend OOF mismatch.")
        result["recomputed_with_official_training_labels"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task2-split-manifest", type=Path, help="optional locally generated official training split")
    parser.add_argument("--task3-labels", type=Path, help="optional official 80-row training coordinates")
    args = parser.parse_args()
    task1 = json.loads(run("task1_plant_seedling", "scripts/verify_results.py"))
    task2_args = ["--split-manifest", args.task2_split_manifest.resolve()] if args.task2_split_manifest else []
    task2 = json.loads(run("task2_facial_expression", "verify_results.py", *task2_args))
    task3 = verify_task3(args.task3_labels.resolve() if args.task3_labels else None)
    task4 = json.loads(run("task4_vessel_segmentation", "scripts/verify_archive.py"))
    fusion2 = task2["pooled"]["Equal_fusion"]
    print(json.dumps({
        "status": "archive_checks_passed",
        "task1": {
            "accuracy": task1["oof"]["accuracy"],
            "macro_f1": task1["oof"]["macro_f1"],
            "checked_training_logs": task1["logged_epochs"],
        },
        "task2": {
            "validation_rows": task2["verified_validation_rows"],
            "training_epoch_rows": task2["verified_epoch_rows"],
            "fusion_accuracy": fusion2["accuracy"],
            "fusion_macro_f1": fusion2["macro_f1"],
            "split_manifest_verified": task2["split_manifest_verified"],
        },
        "task3": task3,
        "task4": {
            "heldout_training_images": task4["validation"]["images"],
            "macro_dice_from_pixel_counts": task4["validation"]["macro_dice"],
            "blend_weights": task4["blend_weights"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
