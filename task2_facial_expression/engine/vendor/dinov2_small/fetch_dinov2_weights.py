"""下载公开的 DINOv2 ViT-S/14 官方 timm 转换权重，固定版本并校验。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPOSITORY = "timm/vit_small_patch14_dinov2.lvd142m"
REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
SHA256 = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
SIZE = 88240510


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "dinov2_vits14.safetensors"
    if target.is_file() and target.stat().st_size == SIZE and digest(target) == SHA256:
        return target
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.safetensors"
    partial = target.with_suffix(".partial")
    start = last_print = time.monotonic()
    received = 0
    print("PHASE weights_download_start", SIZE, flush=True)
    with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            received += len(block)
            if time.monotonic() - last_print >= 10:
                print("PHASE weights_downloading", received, flush=True)
                last_print = time.monotonic()
            if time.monotonic() - start > 180:
                raise TimeoutError("DINOv2 权重下载超过三分钟")
    assert received == SIZE and digest(partial) == SHA256, "权重大小或 SHA256 不符"
    partial.replace(target)
    (directory / "dinov2_vits14.json").write_text(json.dumps({"repository": REPOSITORY,
        "revision": REVISION, "sha256": SHA256, "bytes": SIZE, "pretraining_dataset": "LVD-142M",
        "model_type": "Vision Transformer", "url": url, "local_training_images_uploaded": False},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("PHASE weights_ready", round(time.monotonic() - start, 2), flush=True)
    return target


if __name__ == "__main__":
    print(fetch(Path(__file__).resolve().parent / "pretrained_weights"))
