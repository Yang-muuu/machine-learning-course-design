"""ImageNet HRNet-W18 native-resolution features with a vessel decoder."""
from pathlib import Path
import hashlib
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
import timm

ROOT = Path(__file__).resolve().parent
MODEL = "hrnet_w18.ms_aug_in1k"
REPO = "timm/hrnet_w18.ms_aug_in1k"
REVISION = "7e2c5583769f54514fd87e3ba9de408e33eaba0f"
SHA256 = "f9e3c451448c037d4dd549c96f3013d2926b492f69f27c74bbcc909e053e677c"


@lru_cache(maxsize=1)
def ensure_assets():
    cache = ROOT / "optimization/thirdparty/huggingface"
    cached = cache / "models--timm--hrnet_w18.ms_aug_in1k" / "snapshots" / REVISION / "model.safetensors"
    if not cached.exists():
        from huggingface_hub import hf_hub_download
        cached = Path(hf_hub_download(repo_id=REPO, revision=REVISION, filename="model.safetensors",
            cache_dir=str(cache)))
    if hashlib.sha256(cached.read_bytes()).hexdigest() != SHA256:
        raise ValueError("HRNet checkpoint SHA256 mismatch")
    return cached


def provenance():
    return {"enum": MODEL, "url": f"https://huggingface.co/{REPO}/tree/{REVISION}",
            "sha256": SHA256,
            "dataset": "ImageNet-1K classification; no retinal vessel supervision",
            "license": "MIT", "feature_location": "native multi-resolution branches"}


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))


class HRNetUNet(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(MODEL, pretrained=False, features_only=True, feature_location="")
        if pretrained:
            from safetensors.torch import load_file
            state = load_file(str(ensure_assets()))
            # Keep every native feature tensor and require an exact match.
            # Classification-only heads are deliberately absent from this model.
            selected = {key: state[key] for key in self.encoder.state_dict()}
            self.encoder.load_state_dict(selected, strict=True)
        self.detail = ConvBlock(3, 24)
        widths = [24] + self.encoder.feature_info.channels()
        self.decoders = nn.ModuleList()
        channels = widths[-1]
        for skip, out in zip(reversed(widths[:-1]), [128, 96, 64, 32, 24]):
            self.decoders.append(ConvBlock(channels + skip, out))
            channels = out
        self.head = nn.Conv2d(channels, 1, 1)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.cfg.get("freeze_encoder_bn"):
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, image):
        x = (image - self.mean) / self.std
        features = [self.detail(x)] + self.encoder(x)
        x = features[-1]
        for block, skip in zip(self.decoders, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        return self.head(x)
