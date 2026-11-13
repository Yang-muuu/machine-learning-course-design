"""Unified offline verification entry point for all four course design experiments."""
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


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(folder: str, script: str, *args) -> str:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", script, *map(str, args)],
        cwd=ROOT / folder,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(
            f"{folder}/{script} failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def verify_kaggle_records(csv_dir: Path | None) -> dict:
    records = load(ROOT / "kaggle_submissions.json")["tasks"]
    if [row["task"] for row in records] != [1, 2, 3, 4]:
        raise ValueError("Expected one archived Kaggle record per task.")
    t1_sha = load(ROOT / "task1_plant_seedling/results/summary.json")["source_artifact_sha256"]["original_submission"]
    t2_sha = load(ROOT / "task2_facial_expression/evidence/summary.json")["submissions"]["6"]["sha256"]
    t2_recipe_sha = load(ROOT / "task2_facial_expression/records/kaggle_final_recipe.json")["submission_sha256"]
    if t2_sha != t2_recipe_sha:
        raise ValueError("Task 2 evidence summary and kaggle_final_recipe disagree on submission SHA-256.")
    t3_sha = load(ROOT / "task3_fovea_localization/configs/fusion.json")["submission_sha256"]
    t3_kaggle_sha = load(ROOT / "task3_fovea_localization/records/kaggle_submission.json")["submission_sha256"]
    t3_final_sha = load(ROOT / "task3_fovea_localization/configs/kaggle_final_fusion.json")["submission_sha256"]
    if len({t3_sha, t3_kaggle_sha, t3_final_sha}) != 1:
        raise ValueError("Task 3 fusion recipes and kaggle_submission.json disagree on submission SHA-256.")
    t4_sha = load(ROOT / "task4_vessel_segmentation/records/final_snapshot.json")["submission_sha256"]

    expected_sha = (t1_sha, t2_sha, t3_sha, t4_sha)
    for row, expected in zip(records, expected_sha):
        if row["sha256"] != expected or row["match"] != "byte_identical_sha256" or row["status"] != "COMPLETE":
            raise ValueError(f"Archived Kaggle record disagrees with its task files: task {row['task']}.")
        if csv_dir is not None:
            name = row["local_file"]
            if Path(name).name != name or digest(csv_dir / name) != expected:
                raise ValueError(f"Local CSV checksum mismatch: {name}.")
    return {
        "archived_submission_refs": [row["submission_ref"] for row in records],
        "local_csv_checksums_verified": csv_dir is not None,
    }


def verify_task2() -> dict:
    folder = "task2_facial_expression"
    task = ROOT / folder
    manifest_out = run(folder, "task2.py", "verify").strip()
    summary = load(task / "evidence/summary.json")
    weighted = summary["schemes"]["weighted"]
    stacked = summary["schemes"]["stacked"]
    if weighted["rows"] != 28165 or weighted["correct"] != 20669:
        raise ValueError("Task 2 weighted OOF summary mismatch.")
    if stacked["rows"] != 28165 or stacked["correct"] != 20706:
        raise ValueError("Task 2 stacked OOF summary mismatch.")
    with (task / "evidence/training_curves.csv").open(encoding="utf-8-sig", newline="") as stream:
        curve_rows = list(csv.DictReader(stream))
    expected_models = {"convnext_eca", "dinov2_small", "ddam_drop", "dan", "b2_native", "ddam_old", "ddam_sam"}
    seen_models = {row["model"] for row in curve_rows}
    if seen_models != expected_models or not curve_rows:
        raise ValueError("Task 2 training_curves.csv missing expected architectures.")
    return {
        "manifest_check": manifest_out,
        "base_models_verified": len(seen_models),
        "training_curve_rows": len(curve_rows),
        "scheme_A_weighted_oof_acc": round(weighted["accuracy"], 6),
        "scheme_B_stacked_oof_acc": round(stacked["accuracy"], 6),
        "kaggle_score_A": summary["submissions"]["6"]["kaggle_display_reported"],
        "kaggle_score_B": summary["submissions"]["5"]["kaggle_display_reported"],
        "submission_A_sha256": summary["submissions"]["6"]["sha256"],
        "submission_B_sha256": summary["submissions"]["5"]["sha256"],
    }


def verify_task3(labels: Path | None) -> dict:
    folder = "task3_fovea_localization"
    task = ROOT / folder
    hashes = load(task / "records/training_source_sha256.json")
    if len(hashes) != 14:
        raise ValueError(f"Expected 14 hashed training source files in Task 3, found {len(hashes)}.")
    for name, expected in hashes.items():
        if digest(task / "training" / name) != expected:
            raise ValueError(f"Task 3 source checksum mismatch: {name}")

    plan = load(task / "configs/fusion.json")
    log_rows = 0
    for name, branch in plan["branches"].items():
        expected_epochs = list(range(1, branch["train_epochs"] + 1))
        if branch["folds"] == 5:
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
        elif branch["folds"] == 1:
            with (task / "records" / name / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if [int(row["epoch"]) for row in rows] != expected_epochs:
                raise ValueError(f"Incomplete Task 3 full-data log: {name}.")
            replay = load(task / "records" / name / "replay_verification.json")
            if replay.get("status") != "PASS":
                raise ValueError(f"Task 3 replay verification did not pass for {name}.")
            log_rows += len(rows)
        else:
            raise ValueError(f"Unexpected fold count for {name}: {branch['folds']}")

    with tempfile.TemporaryDirectory() as temp:
        rebuilt_final = Path(temp) / "submission.csv"
        run(folder, "tools/rebuild_submission.py", "--recipe", "configs/fusion.json", "--verify", "--output", rebuilt_final)
        if digest(rebuilt_final) != digest(task / "predictions/submission.csv"):
            raise ValueError("Task 3 rebuilt final submission differs from archived predictions/submission.csv.")
        rebuilt_oof = Path(temp) / "oof_candidate.csv"
        run(folder, "tools/rebuild_submission.py", "--recipe", "configs/oof_candidate_fusion.json", "--verify", "--output", rebuilt_oof)

    summary = load(task / "records/oof_summary.json")
    result = {
        "branches_verified": len(plan["branches"]),
        "epoch_log_rows_checked": log_rows,
        "source_files_checked": len(hashes),
        "rebuilt_submission_rows": 40,
        "final_submission_sha256": plan["submission_sha256"],
        "archived_r30_v20_oof_rmse_px": summary["equal_fusion"]["oof_rmse"],
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
        if not isclose(actual["equal_fusion"]["oof_rmse"], result["archived_r30_v20_oof_rmse_px"], abs_tol=1e-8):
            raise ValueError("Task 3 equal_fusion OOF mismatch.")
        result["recomputed_with_official_training_labels"] = True
        result["recomputed_final_y_axis_oof_rmse_px"] = actual["final_y_axis_fusion"]["y_rmse"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task3-labels", type=Path, help="optional official 80-row training coordinates for Task 3")
    parser.add_argument("--kaggle-csv-dir", type=Path, help="optional directory containing the four submitted CSV files")
    args = parser.parse_args()

    kaggle = verify_kaggle_records(args.kaggle_csv_dir.resolve() if args.kaggle_csv_dir else None)
    task1 = json.loads(run("task1_plant_seedling", "scripts/verify_results.py"))
    task2 = verify_task2()
    task3 = verify_task3(args.task3_labels.resolve() if args.task3_labels else None)
    task4 = json.loads(run("task4_vessel_segmentation", "scripts/verify_archive.py"))

    print(
        json.dumps(
            {
                "status": "all_four_tasks_verified",
                "kaggle": kaggle,
                "task1": {
                    "accuracy": task1["oof"]["accuracy"],
                    "macro_f1": task1["oof"]["macro_f1"],
                    "checked_training_logs": task1["logged_epochs"],
                },
                "task2": task2,
                "task3": task3,
                "task4": {
                    "heldout_training_images": task4["validation"]["images"],
                    "macro_dice_from_pixel_counts": task4["validation"]["macro_dice"],
                    "blend_weights": task4["blend_weights"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
