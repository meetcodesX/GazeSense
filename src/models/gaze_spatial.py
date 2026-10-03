"""Spatial-preserving gaze models (heatmap output + soft-argmax).

Why: the baseline CNN ends with AdaptiveAvgPool2d(1,1), which throws away
*where* things are. Here the scene feature map stays spatial (7x7 for a
224 input), is decoded back up to a HxH heatmap, and the gaze point is read
out with a soft-argmax.

variant="scene_mask"   : scene + head mask only (no head crop)   -> ablation
variant="two_pathway"  : scene + head mask  AND  head-crop pathway fused
                         into the scene feature map               -> main model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _resnet18(pretrained):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return models.resnet18(weights=weights)


class SceneEncoder(nn.Module):
    """ResNet18 up to layer4, input = RGB + head mask (4 channels)."""

    def __init__(self, pretrained=True):
        super().__init__()
        r = _resnet18(pretrained)
        old = r.conv1
        conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3] = old.weight
            conv1.weight[:, 3:] = 0.0  # mask channel starts neutral
        self.stem = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool)
        self.layers = nn.Sequential(r.layer1, r.layer2, r.layer3, r.layer4)

    def forward(self, x):
        return self.layers(self.stem(x))  # (B, 512, S/32, S/32)


class HeadEncoder(nn.Module):
    """ResNet18 on the head crop -> one 512-d vector."""

    def __init__(self, pretrained=True):
        super().__init__()
        r = _resnet18(pretrained)
        self.body = nn.Sequential(
            r.conv1, r.bn1, r.relu, r.maxpool,
            r.layer1, r.layer2, r.layer3, r.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.body(x)).flatten(1)


def _up(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class GazeSpatialModel(nn.Module):
    def __init__(self, variant="two_pathway", pretrained=True, heatmap_size=56):
        super().__init__()
        assert variant in ("scene_mask", "two_pathway")
        self.variant = variant
        self.H = heatmap_size

        self.scene = SceneEncoder(pretrained)

        if variant == "two_pathway":
            self.head = HeadEncoder(pretrained)
            self.head_proj = nn.Linear(512, 256)
            fuse_in = 512 + 256
        else:
            self.head = None
            fuse_in = 512

        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 7 -> 14 -> 28 -> 56
        self.decoder = nn.Sequential(_up(256, 128), _up(128, 64), _up(64, 32))
        self.out = nn.Conv2d(32, 1, 1)

        # learnable inverse temperature for the soft-argmax
        self.beta = nn.Parameter(torch.tensor(1.0))

        grid = (torch.arange(heatmap_size, dtype=torch.float32) + 0.5) / heatmap_size
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, scene, head_mask, head_crop=None):
        x = torch.cat([scene, head_mask], dim=1)
        feat = self.scene(x)  # (B, 512, h, w)

        if self.variant == "two_pathway":
            hv = F.relu(self.head_proj(self.head(head_crop)))  # (B, 256)
            hv = hv[:, :, None, None].expand(-1, -1, feat.size(2), feat.size(3))
            feat = torch.cat([feat, hv], dim=1)

        feat = self.fuse(feat)
        logits = self.out(self.decoder(feat))  # (B, 1, H, H)
        if logits.shape[-1] != self.H:
            logits = F.interpolate(
                logits, size=(self.H, self.H), mode="bilinear", align_corners=False
            )
        return logits

    def soft_argmax(self, logits):
        b = logits.size(0)
        p = F.softmax(self.beta * logits.view(b, -1), dim=1).view(b, self.H, self.H)
        x = (p.sum(1) * self.grid).sum(1)
        y = (p.sum(2) * self.grid).sum(1)
        return torch.stack([x, y], dim=1)

    def hard_argmax(self, logits):
        b = logits.size(0)
        idx = logits.view(b, -1).argmax(1)
        y = (idx // self.H).float()
        x = (idx % self.H).float()
        return torch.stack([(x + 0.5) / self.H, (y + 0.5) / self.H], dim=1)


if __name__ == "__main__":
    for v in ("scene_mask", "two_pathway"):
        m = GazeSpatialModel(v, pretrained=False)
        lg = m(torch.randn(2, 3, 224, 224), torch.rand(2, 1, 224, 224), torch.randn(2, 3, 112, 112))
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(v, tuple(lg.shape), tuple(m.soft_argmax(lg).shape), f"{n:.1f}M params")
