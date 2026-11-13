"""表情分类数据审计。支持原始 ZIP 或解压目录；不依赖 PyTorch。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from collections import Counter
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

# 必须显式定义题目编号：ImageFolder 的字母序会把 Neutral 错排为 3。
CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class DataSource:
    """在 ZIP 内直接读取，避免解压数万张小图片；目录输入也使用相同逻辑。"""

    def __init__(self, source: str | Path):
        self.path = Path(source).expanduser().resolve()
        self.archive = None
        if self.path.is_file() and zipfile.is_zipfile(self.path):
            self.archive = zipfile.ZipFile(self.path)
            self.names = sorted(n for n in self.archive.namelist() if not n.endswith("/"))
            if len(self.names) != len(set(self.names)):
                raise ValueError("ZIP 有重复条目名称，请先修复数据包。")
        elif self.path.is_dir():
            self.names = sorted(p.relative_to(self.path).as_posix()
                                for p in self.path.rglob("*") if p.is_file())
        else:
            raise FileNotFoundError(f"数据路径不存在或不是 ZIP/目录：{self.path}")

    def read(self, name: str) -> bytes:
        if self.archive is not None:
            return self.archive.read(name)
        return (self.path / name).read_bytes()

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()

    def fingerprint(self) -> str:
        """缓存按数据内容标识；ZIP 用整个包的 SHA256，目录用路径、大小和 mtime。"""
        digest = hashlib.sha256()
        if self.archive is not None:
            with self.path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            for name in self.names:
                stat = (self.path / name).stat()
                digest.update(f"{name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
        return digest.hexdigest()


def read_image(raw: bytes) -> tuple[np.ndarray, str]:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()  # 真正解码，而不是仅检查文件后缀。
        if image.size != (48, 48):
            raise ValueError(f"期望 48x48，实际为 {image.size}")
        mode = image.mode
        pixels = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    return pixels, mode


def prepare_data(source: str | Path, cache: str | Path, refresh: bool = False) -> dict:
    """检查坏图、标签冲突、重复图、模板 ID，并保存训练/测试数组及审计报告。"""
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    data_source = DataSource(source)
    try:
        fingerprint = data_source.fingerprint()
        audit_path = cache / "data_audit.json"
        required = ["train_images.npy", "train_manifest.csv", "test_images.npy",
                    "test_manifest.csv", "sample_submission.csv", "data_audit.json"]
        if not refresh and all((cache / name).exists() for name in required):
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("fingerprint") == fingerprint and audit.get("schema_version") == 1:
                return audit

        csv_names = [n for n in data_source.names
                     if Path(n).name.lower() in {"submission.csv", "sample_submission.csv"}]
        if len(csv_names) != 1:
            raise ValueError(f"应恰好找到一个提交模板，实际找到：{csv_names}")
        sample = pd.read_csv(io.BytesIO(data_source.read(csv_names[0])), dtype={"ID": str})
        if sample.columns.tolist() != ["ID", "Emotion"]:
            raise ValueError(f"本题模板列必须是 ID,Emotion，实际是 {sample.columns.tolist()}")
        if sample.empty or sample["ID"].isna().any() or sample["ID"].duplicated().any():
            raise ValueError("模板为空、ID 缺失或 ID 重复。")

        train_records, train_pixels, test_paths = [], [], {}
        bad_images, ignored, mode_counts = [], [], Counter()
        raw_class_counts = Counter()
        for name in data_source.names:
            parts = Path(name).parts
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if "train" in parts:
                index = parts.index("train")
                if len(parts) != index + 3 or parts[index + 1] not in CLASS_TO_LABEL:
                    raise ValueError(f"未知训练类或目录层级：{name}")
                emotion = parts[index + 1]
                raw_class_counts[emotion] += 1
                try:
                    pixels, mode = read_image(data_source.read(name))
                except (OSError, ValueError) as exc:
                    # 坏训练图可隔离，但必须留下路径和原因，不能默默丢弃。
                    bad_images.append({"path": name, "reason": str(exc)})
                    continue
                mode_counts[mode] += 1
                train_pixels.append(pixels)
                train_records.append({"path": name, "ID": Path(name).name,
                                      "label": CLASS_TO_LABEL[emotion], "class_name": emotion,
                                      "pixel_hash": hashlib.sha256(pixels.tobytes()).hexdigest(),
                                      "pixel_std": float(pixels.std())})
            elif "test" in parts:
                image_id = Path(name).name
                if image_id in test_paths:
                    raise ValueError(f"测试集重复文件名：{image_id}")
                test_paths[image_id] = name
            else:
                ignored.append(name)

        expected_ids = sample["ID"].tolist()
        if set(test_paths) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(test_paths))[:10]
            extra = sorted(set(test_paths) - set(expected_ids))[:10]
            raise ValueError(f"模板与测试文件集合不同。缺少：{missing}；多出：{extra}")
        if not train_records:
            raise ValueError("没有找到有效训练图像。请传包含 train/test/提交模板的目录。")

        train = pd.DataFrame(train_records)
        pixels_array = np.stack(train_pixels)
        # 同一灰度图有不同标签，无法建立唯一监督目标；整个冲突组都隔离。
        conflicts = train.groupby("pixel_hash")["label"].nunique()
        conflict_hashes = set(conflicts[conflicts > 1].index)
        quarantine = train[train["pixel_hash"].isin(conflict_hashes)].copy()
        keep = ~train["pixel_hash"].isin(conflict_hashes)
        train = train.loc[keep].reset_index(drop=True)
        pixels_array = pixels_array[keep.to_numpy()]
        if set(train["label"]) != set(range(6)):
            raise ValueError("清洗后未包含全部六个类别。")
        train["array_index"] = np.arange(len(train))

        test_records, test_pixels = [], []
        for image_id in expected_ids:  # 模板顺序与文件系统排序不同，必须以模板为准。
            name = test_paths[image_id]
            try:
                pixels, mode = read_image(data_source.read(name))
            except (OSError, ValueError) as exc:
                raise ValueError(f"测试图不能跳过：{name}；{exc}") from exc
            mode_counts[mode] += 1
            test_pixels.append(pixels)
            test_records.append({"path": name, "ID": image_id,
                                 "pixel_hash": hashlib.sha256(pixels.tobytes()).hexdigest(),
                                 "pixel_std": float(pixels.std())})
        test = pd.DataFrame(test_records)
        counts = train["pixel_hash"].value_counts()
        audit = {
            "schema_version": 1, "source": str(data_source.path), "fingerprint": fingerprint,
            "class_to_label": CLASS_TO_LABEL,
            "raw_train_count": len(train_records) + len(bad_images),
            "raw_class_counts": dict(raw_class_counts),
            "clean_train_count": len(train), "test_count": len(test),
            "clean_class_counts": {name: int((train["label"] == label).sum())
                                   for name, label in CLASS_TO_LABEL.items()},
            "bad_train_images": bad_images,
            "conflicting_hash_groups": len(conflict_hashes),
            "quarantined_train_count": len(quarantine),
            "same_label_duplicate_groups": int((counts > 1).sum()),
            "same_label_duplicate_excess_images": int((counts - 1).clip(lower=0).sum()),
            "test_duplicate_excess_images": int(test["pixel_hash"].duplicated().sum()),
            "train_test_shared_hashes": len(set(train["pixel_hash"]) & set(test["pixel_hash"])),
            "low_variance_train_images_std_lt_2": int((train["pixel_std"] < 2).sum()),
            "low_variance_test_images_std_lt_2": int((test["pixel_std"] < 2).sum()),
            "decoded_image_modes": dict(mode_counts), "ignored_images": ignored,
            "image_shape": [48, 48], "template_columns": sample.columns.tolist(),
            "test_order": "original submission.csv ID order",
            "overlap_policy": "audit only; never copy training labels to test predictions",
        }
        np.save(cache / "train_images.npy", pixels_array)
        np.save(cache / "test_images.npy", np.stack(test_pixels))
        train.to_csv(cache / "train_manifest.csv", index=False)
        test.to_csv(cache / "test_manifest.csv", index=False)
        quarantine.to_csv(cache / "quarantined_label_conflicts.csv", index=False)
        pd.DataFrame(bad_images, columns=["path", "reason"]).to_csv(cache / "bad_train_images.csv", index=False)
        sample.to_csv(cache / "sample_submission.csv", index=False)
        write_json(audit_path, audit)
        return audit
    finally:
        data_source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="原始 ZIP 或包含完整数据的解压目录")
    parser.add_argument("--cache", default="data_cache")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.data, args.cache, args.refresh), ensure_ascii=False, indent=2))
