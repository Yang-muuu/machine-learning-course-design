"""RETFound-Green ViT-S/14 plus a newly initialized vessel decoder.

Encoder weights: Engelmann & Bernabeu, RETFound-Green v0.1.
https://github.com/justinengelmann/RETFound_Green
The encoder and derived weights retain RETFOUND_GREEN_LICENSE.txt.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F
import timm
from torchvision import models
from hrnet_adapter import ConvBlock

ROOT = Path(__file__).resolve().parent
URL = "https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth"
SHA256 = "431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1"
SIZE = 87396676


def use_group_norm(module):
    """Replace newly initialized decoder BatchNorm without touching encoders."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = min(8, child.num_features)
            while child.num_features % groups:
                groups -= 1
            norm = nn.GroupNorm(groups, child.num_features, eps=child.eps)
            with torch.no_grad():
                norm.weight.copy_(child.weight)
                norm.bias.copy_(child.bias)
            setattr(module, name, norm)
        else:
            use_group_norm(child)


@lru_cache(maxsize=1)
def ensure_assets():
    path = ROOT / "optimization/thirdparty/retfoundgreen/retfoundgreen_statedict.pth"
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
                    raise ValueError("Invalid RETFound-Green resume response")
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
        if partial.stat().st_size != SIZE or hashlib.sha256(partial.read_bytes()).hexdigest() != SHA256:
            raise ValueError("RETFound-Green download verification failed")
        partial.replace(path)
    if path.stat().st_size != SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("RETFound-Green checkpoint identity mismatch")
    return path


def provenance(cfg=None):
    result = {"enum": "RETFound-Green v0.1 ViT-S/14 reg4", "url": URL, "sha256": SHA256,
            "dataset": "75,000 public fundus photos from AIROGS, DDR and ODIR; self-supervised token reconstruction",
            "supervision": "no released vessel head; this experiment trains a new decoder on each fold's 16 images",
            "normalization": "RGB [0,1] -> [-1,1], official mean/std .5; dynamic positions from official 392 grid",
            "license": "author CNCRL non-commercial research; RETFOUND_GREEN_LICENSE.txt"}
    if cfg and cfg.get("task") == "external_pretrain":
        result["supervision"] = "public foundation encoder; dense vessel head learned only from the external FIVES training split"
    if cfg and cfg.get("cnn_skip") == "resnet18":
        result["local_feature_pretraining"] = {"enum": "ResNet18 IMAGENET1K_V1 stem/layer1/layer2",
            "url": models.ResNet18_Weights.IMAGENET1K_V1.url, "dataset": "ImageNet-1K classification"}
    if cfg and cfg.get("cnn_skip") == "vascx":
        from vascx_features import provenance as local_provenance
        result["local_feature_pretraining"] = local_provenance(int(cfg.get("vascx_member", 0)))
        result["local_feature_scale"] = cfg.get("vessel_scale", 1)
    return result


class RETFoundUNet(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=False,
            img_size=(392, 392), num_classes=0, dynamic_img_size=True)
        if pretrained:
            state = torch.load(ensure_assets(), map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state, strict=True)
        self.detail = ConvBlock(3, 16)
        self.use_cnn_skip = cfg.get("cnn_skip") == "resnet18"
        self.use_vessel_skip = cfg.get("cnn_skip") == "vascx"
        if self.use_vessel_skip:
            from vascx_features import VesselLocalEncoder
            self.vessel_encoder = VesselLocalEncoder(pretrained, int(cfg.get("vascx_member", 0)))
            skip_channels = [128, 96, 64, 48]
        elif self.use_cnn_skip:
            cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.cnn_encoder = nn.ModuleList([nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu),
                                             nn.Sequential(cnn.maxpool, cnn.layer1), cnn.layer2])
            self.register_buffer("cnn_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
            self.register_buffer("cnn_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
            skip_channels = [128, 64, 64, 16]
        else:
            self.half_detail = ConvBlock(16, 32)
            self.quarter_detail = ConvBlock(32, 48)
            self.eighth_detail = ConvBlock(48, 64)
            skip_channels = [64, 48, 32, 16]
        self.token_projections = nn.ModuleList([nn.Conv2d(384, 64, 1) for _ in range(4)])
        self.token_fuse = ConvBlock(256, 128)
        self.decoders = nn.ModuleList()
        width = 128
        for skip, next_width in zip(skip_channels, [96, 64, 32, 24]):
            self.decoders.append(ConvBlock(width + skip, next_width))
            width = next_width
        self.dropout = nn.Dropout2d(float(cfg.get("dropout", .05)))
        self.head = nn.Conv2d(24, 1, 1)
        if cfg.get("decoder_norm") == "group":
            new_modules = [self.detail, self.token_fuse, self.decoders]
            if not self.use_cnn_skip and not self.use_vessel_skip:
                new_modules += [self.half_detail, self.quarter_detail, self.eighth_detail]
            for module in new_modules:
                use_group_norm(module)
        if cfg.get("freeze_encoder"):
            for parameter in self.pretrained_parameters():
                parameter.requires_grad_(False)
        elif cfg.get("freeze_foundation"):
            self.encoder.requires_grad_(False)

    def pretrained_parameters(self):
        local = list(self.vessel_encoder.parameters()) if self.use_vessel_skip else list(self.cnn_encoder.parameters()) if self.use_cnn_skip else []
        return list(self.encoder.parameters()) + local

    def train(self, mode=True):
        super().train(mode)
        if mode and (self.cfg.get("freeze_encoder") or self.cfg.get("freeze_foundation")):
            self.encoder.eval()
            if self.use_cnn_skip and self.cfg.get("freeze_encoder"):
                self.cnn_encoder.eval()
            if self.use_vessel_skip and self.cfg.get("freeze_encoder"):
                self.vessel_encoder.eval()
        return self

    def forward(self, image):
        if image.shape[-1] % 14 or image.shape[-2] % 14:
            raise ValueError("RETFound-Green input dimensions must be divisible by patch size 14")
        rgb = image[:, :3]
        x = rgb * 2 - 1
        full = self.detail(x)
        if self.use_vessel_skip:
            scale = float(self.cfg.get("vessel_scale", 1))
            local_input = image if scale == 1 else F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
            local_features = self.vessel_encoder(local_input)
            if scale != 1:
                height, width = image.shape[-2:]
                local_features = [F.interpolate(value, size=(height // 2 ** i, width // 2 ** i),
                    mode="bilinear", align_corners=False) for i, value in enumerate(local_features)]
            full = torch.cat([full, local_features[0]], 1)
            half, quarter, eighth = local_features[1:]
        elif self.use_cnn_skip:
            half = self.cnn_encoder[0]((rgb - self.cnn_mean) / self.cnn_std)
            quarter = self.cnn_encoder[1](half)
            eighth = self.cnn_encoder[2](quarter)
        else:
            half = self.half_detail(F.avg_pool2d(full, 2))
            quarter = self.quarter_detail(F.avg_pool2d(half, 2))
            eighth = self.eighth_detail(F.avg_pool2d(quarter, 2))
        tokens = self.encoder.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True,
                                                     return_prefix_tokens=False, norm=True)
        x = self.token_fuse(torch.cat([layer(feature) for layer, feature in zip(self.token_projections, tokens)], 1))
        for decoder, skip in zip(self.decoders, [eighth, quarter, half, full]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], 1))
        return self.head(self.dropout(x))
