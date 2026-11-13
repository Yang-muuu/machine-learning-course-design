"""下载 timm 发布的 ImageNet 权重；固定提交及 SHA256，不读取测试答案。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
MODELS = ("mobilevit_xs", "mobilevitv2_100")
PINNED = {
    "mobilevit_xs": {"revision": "7a989ba56f36ed9f230044306e1e31563132c35e",
                     "bytes": 9336550, "sha256": "0dab21a2752b2418a9db1486a4c03488569b4b5eca7d4197784d030abbce801f"},
    "mobilevitv2_100": {"revision": "8d5eb1ea9e84a61f28fce38ea79043e7d9a75a9c",
                       "bytes": 19697308, "sha256": "103bf57c1001cb735f72ad4632c3ec1790005d9535f952a7823422cdb32d401e"},
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch(architecture, directory):
    assert architecture in MODELS
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{architecture}.safetensors"
    manifest_path = directory / f"{architecture}.json"
    pinned = PINNED[architecture]
    if manifest_path.is_file() and target.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["sha256"] == pinned["sha256"] and manifest["revision"] == pinned["revision"]
                and sha256(target) == pinned["sha256"]):
            return target, manifest
    repo = f"timm/{architecture}.cvnets_in1k"
    # 提交、大小和哈希来自 2026-10-12 官方仓库 API；后续运行不随仓库最新版本变化。
    expected, revision = pinned["sha256"], pinned["revision"]
    assert len(expected) == 64 and len(revision) == 40
    url = f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors"
    partial = target.with_suffix(".partial")
    start, received, last_print = time.monotonic(), 0, time.monotonic()
    print(f"PHASE download_start architecture={architecture} bytes={pinned['bytes']}", flush=True)
    with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            received += len(block)
            if time.monotonic() - last_print > 10:
                print(f"PHASE downloading architecture={architecture} bytes={received}", flush=True)
                last_print = time.monotonic()
            if time.monotonic() - start > 180:
                raise TimeoutError("预训练权重下载超过三分钟")
    assert received == pinned["bytes"] and sha256(partial) == expected, "官方权重大小/SHA256 不匹配"
    partial.replace(target)
    manifest = {"architecture": architecture, "repository": repo, "revision": revision,
                "url": url, "bytes": received, "sha256": expected,
                "pretraining_dataset": "ImageNet-1k", "source": "paper authors / timm conversion"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PHASE download_complete architecture={architecture} seconds={time.monotonic()-start:.1f}", flush=True)
    return target, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--directory", type=Path, default=ROOT / "pretrained_weights")
    args = parser.parse_args()
    for architecture in args.models:
        path, manifest = fetch(architecture, args.directory)
        print(json.dumps({"path": str(path), **manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
