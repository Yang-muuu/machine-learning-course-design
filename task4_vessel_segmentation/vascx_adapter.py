"""Binary-vessel VascX transfer with the authors' crop and contrast pipeline."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

try:
    import cv2
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / ".runtime"))
    import cv2

from rtnls_fundusprep.mask_extraction import get_cfi_bounds

ROOT = Path(__file__).resolve().parent
URL = "https://media.githubusercontent.com/media/Eyened/retinalysis-vascx-models/main/vessels/vessels_july24.pt"
SIZE = 352821632
SHA256 = "bdabae77502648acd1c176bff6d5e3c8295da60f18f2f84fe1bcd6181d2b2ca4"


def ensure_assets():
    path = ROOT / "optimization" / "thirdparty" / "vascx" / "vessels_july24.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(".download")
        for _ in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == SIZE:
                break
            request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Invalid VascX resume response")
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
        if partial.stat().st_size != SIZE or hashlib.sha256(partial.read_bytes()).hexdigest() != SHA256:
            raise ValueError("VascX download failed size/SHA256 verification")
        partial.replace(path)
    if path.stat().st_size != SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("VascX checkpoint failed size/SHA256 verification")
    return path


def provenance():
    return {"enum": "VascX binary vessels_july24", "url": URL, "sha256": SHA256,
            "dataset": "CHASEDB, DRHAGIS, FIVES, HRF, LEUVEN, RETA, RS; confirmed embedded dataframe_path",
            "source_repository": "https://github.com/Eyened/retinalysis-vascx-models",
            "normalization": "cropped RGB + mirrored contrast-enhanced RGB; ImageNet normalization on both",
            "preprocessing_source": "rtnls_fundusprep/UPSTREAM.json",
            "license": "weights CC-BY-NC-4.0; fundus preprocessing AGPL-3.0"}


@lru_cache(maxsize=32)
def prepare_image(path, size=1024):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    height, width = rgb.shape[:2]
    bounds = get_cfi_bounds(rgb)
    transform, cropped_bounds = bounds.crop(size)
    cropped = transform.warp(rgb)
    contrast = cropped_bounds.make_contrast_enhanced_res256(image=cropped, sigma_fraction=.05)
    x = np.concatenate([cropped, contrast], -1).transpose(2, 0, 1).astype(np.float32) / 255
    # Pixel centers of the fixed PIL-resized 512 grid expressed in raw pixels.
    canonical_to_raw = np.array([[width / 512, 0, (width / 512 - 1) / 2],
                                 [0, height / 512, (height / 512 - 1) / 2], [0, 0, 1]])
    to_model = transform.M @ canonical_to_raw
    geometry = {"canonical_to_model": to_model.tolist(),
                "model_to_canonical": np.linalg.inv(to_model).tolist(),
                "raw_to_model": transform.M.tolist(), "raw_height": height, "raw_width": width,
                "bounds": bounds.to_dict(), "size": size}
    return np.ascontiguousarray(x), geometry


def warp_label(canonical_label, geometry):
    size = int(geometry["size"])
    return cv2.warpPerspective(canonical_label.astype(np.float32),
        np.asarray(geometry["canonical_to_model"]), (size, size), flags=cv2.INTER_NEAREST)


def restore_probability(probability, geometry):
    restored = cv2.warpPerspective(probability.astype(np.float32),
        np.asarray(geometry["model_to_canonical"]), (512, 512), flags=cv2.INTER_LINEAR)
    if not np.isfinite(restored).all():
        raise FloatingPointError("Nonfinite restored vessel probabilities")
    return restored.clip(0, 1)


def restore_native_probability(probability, geometry):
    native = cv2.warpPerspective(probability.astype(np.float32),
        np.linalg.inv(np.asarray(geometry["raw_to_model"])),
        (int(geometry["raw_width"]), int(geometry["raw_height"])), flags=cv2.INTER_LINEAR)
    return native.clip(0, 1)


def restore_native_first(probability, geometry):
    native = restore_native_probability(probability, geometry)
    restored = np.asarray(Image.fromarray(native).resize((512, 512), Image.Resampling.NEAREST), dtype=np.float32)
    if not np.isfinite(restored).all():
        raise FloatingPointError("Nonfinite native-first probabilities")
    return restored.clip(0, 1)


@lru_cache(maxsize=32)
def prepare_resize(path, size=1024):
    """Fine-tuning variant: preserve the canonical rectangular image frame."""
    rgb = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
    bounds = get_cfi_bounds(rgb)
    contrast = bounds.make_contrast_enhanced_res256(image=rgb, sigma_fraction=.05)
    return np.ascontiguousarray(np.concatenate([rgb, contrast], -1).transpose(2, 0, 1), dtype=np.float32) / 255


@lru_cache(maxsize=32)
def prepare_native(path, padded_size=640, center=False):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    height, width = rgb.shape[:2]
    if max(height, width) > padded_size:
        raise ValueError("Native VascX input exceeds declared padding size")
    bounds = get_cfi_bounds(rgb)
    contrast = bounds.make_contrast_enhanced_res256(image=rgb, sigma_fraction=.05)
    image = np.concatenate([rgb, contrast], -1).transpose(2, 0, 1).astype(np.float32) / 255
    delta_h, delta_w = padded_size - height, padded_size - width
    top, left = (delta_h // 2, delta_w // 2) if center else (0, 0)
    return np.ascontiguousarray(np.pad(image, ((0, 0), (top, delta_h - top), (left, delta_w - left))))


class VascXAdapter(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        # A Python file handle avoids libtorch's narrow Windows path handling.
        with ensure_assets().open("rb") as stream:
            release = torch.jit.load(stream, map_location="cpu").eval()
        member = cfg.get("vascx_member", 0)
        self.all_members = member == "all"
        self.network = release if self.all_members else getattr(release, f"fold{int(member)}").model
        # The graph is supplied by the hash-pinned release even when an adapted
        # tensor checkpoint will immediately replace its parameters.
        self.register_buffer("mean", torch.tensor([.485, .456, .406] * 2).view(1, 6, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225] * 2).view(1, 6, 1, 1))

    @property
    def encoder(self):
        return self.network

    @property
    def head(self):
        network = self.network.fold0.model if self.all_members else self.network
        return network.output_block.conv.conv

    def forward(self, image):
        logits = self.network((image - self.mean) / self.std)
        if self.all_members:
            logits = logits.mean(0)
        return logits[:, 1:2] - logits[:, 0:1]


@torch.inference_mode()
def predict_author_pipeline(model, image, device, tile_batch=1, tta=True):
    """Average sliding-window and flip logits before sigmoid, as upstream does."""
    model.eval()
    x = torch.from_numpy(image[None]).to(device)
    height, width = image.shape[-2:]
    patch, stride = 512, 256
    ys, xs = list(range(0, height - patch + 1, stride)), list(range(0, width - patch + 1, stride))
    if ys[-1] != height - patch:
        ys.append(height - patch)
    if xs[-1] != width - patch:
        xs.append(width - patch)
    positions = [(y, xx) for y in ys for xx in xs]
    axis = torch.arange(patch, device=device, dtype=torch.float32) - (patch - 1) / 2
    gaussian = torch.exp(-.5 * (axis / (patch * .125)).square())
    window = (gaussian[:, None] * gaussian[None, :]).clamp_min(.001)[None, None]
    flip_results = []
    for dims in ([(), (-1,), (-2,), (-2, -1)] if tta else [()]):
        transformed = x.flip(dims) if dims else x
        total = torch.zeros(1, 1, height, width, device=device)
        weight = torch.zeros_like(total)
        for start in range(0, len(positions), tile_batch):
            subset = positions[start:start + tile_batch]
            tiles = torch.cat([transformed[:, :, y:y + patch, xx:xx + patch] for y, xx in subset])
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(tiles)
            for logit, (y, xx) in zip(logits.float(), subset):
                total[:, :, y:y + patch, xx:xx + patch] += logit[None] * window
                weight[:, :, y:y + patch, xx:xx + patch] += window
        result = total / weight
        flip_results.append(result.flip(dims) if dims else result)
    return torch.stack(flip_results).mean(0).sigmoid()[0, 0].cpu().numpy()
