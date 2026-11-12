"""Native PyTorch replicas of four public VascX vessel encoder blocks."""
import torch
from torch import nn
from torch.nn import functional as F
from vascx_adapter import ensure_assets, provenance as vascx_provenance


class VesselBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True, track_running_stats=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True, track_running_stats=False)
        self.act = nn.LeakyReLU(.01, inplace=True)

    def forward(self, image):
        return self.act(self.norm2(self.conv2(self.act(self.norm1(self.conv1(image))))))


class VesselLocalEncoder(nn.Module):
    def __init__(self, pretrained=True, member=0):
        super().__init__()
        self.blocks = nn.ModuleList([VesselBlock(6, 32, 1), VesselBlock(32, 64, 2),
                                     VesselBlock(64, 96, 2), VesselBlock(96, 128, 2)])
        self.register_buffer("mean", torch.tensor([.485, .456, .406] * 2).view(1, 6, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225] * 2).view(1, 6, 1, 1))
        if pretrained:
            with ensure_assets().open("rb") as stream:
                source = torch.jit.load(stream, map_location="cpu").eval()
            cursor = getattr(source, f"fold{member}").model.skip_layers
            for block in self.blocks:
                # The active skip-layer reference retains forward methods;
                # unused aliases in the traced archive retain only parameters.
                values = {key.replace(".conv.", "."): value for key, value in cursor.downsample.state_dict().items()}
                block.load_state_dict(values, strict=True)
                cursor = cursor.next_layer

    def forward(self, image):
        if image.shape[1] != 6:
            raise ValueError("VascX local features require native RGB plus contrast-enhanced RGB")
        x, output = (image - self.mean) / self.std, []
        for block in self.blocks:
            x = block(x)
            output.append(x)
        return output


def provenance(member=0):
    return {"initialization": vascx_provenance(), "external_member": member,
            "layers": "input block and first three downsampling blocks; native Conv/InstanceNorm/LeakyReLU replica"}


class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True))


class VesselLocalUNet(nn.Module):
    """A compact vessel-specialized encoder with a newly trained dense decoder."""
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.encoder = VesselLocalEncoder(pretrained, int(cfg.get("vascx_member", 0)))
        self.decoders = nn.ModuleList([DecoderBlock(128 + 96, 96), DecoderBlock(96 + 64, 64),
                                      DecoderBlock(64 + 32, 32)])
        self.dropout = nn.Dropout2d(float(cfg.get("dropout", .05)))
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, image):
        features = self.encoder(image)
        x = features[-1]
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], 1))
        return self.head(self.dropout(x))
