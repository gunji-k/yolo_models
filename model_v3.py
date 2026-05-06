"""YOLOv3 with ImageNet-pretrained ResNet-50 backbone (arXiv 1804.02767).

ResNet-50 is the closest off-the-shelf analogue of Darknet-53 used in the
paper (~77% ImageNet top-1, residual, comparable FLOPs); ResNet-18 was used
earlier for a head-to-head architectural A/B with v2 but caps the mAP gain.

Differences vs the v2 model:
- Three detection heads at strides 8, 16, 32 (FPN-style top-down + lateral).
- 9 anchors total, 3 per scale (smallest at stride 8, largest at stride 32).
- Returns a list of 3 tensors [N, Si, Si, 3, 5+C] (one per scale).
- Class branch is per-anchor independent logits (BCE applied in loss).
- Box parameterization (sigmoid xy, exp wh) is identical to v2.

Anchors are stored as normalized [0, 1] (image-relative) (w, h) pairs grouped
into 3 lists of 3. The model/loss/decoder multiply by S at use time, so the
same anchor set is valid under multi-scale training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Default 9 anchors: v3 paper COCO priors / 416, sorted by area into 3 scales.
# Replace via train_v3.py --anchors after running anchors_kmeans.py with k=9.
DEFAULT_ANCHORS = [
    # stride 8 head (smallest objects)
    [(0.0240, 0.0313), (0.0385, 0.0721), (0.0793, 0.0553)],
    # stride 16 head
    [(0.0721, 0.1466), (0.1490, 0.1082), (0.1418, 0.2861)],
    # stride 32 head (largest objects)
    [(0.2788, 0.2163), (0.3750, 0.4760), (0.8966, 0.7837)],
]


class CNNBlock(nn.Module):
    def __init__(self, in_c, out_c, **kw):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, bias=False, **kw)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FPNBlock(nn.Module):
    """5-conv reduce/expand block. Output (mid_c channels) feeds both the
    prediction head at this scale and the top-down route to the next scale."""

    def __init__(self, in_c, mid_c):
        super().__init__()
        self.block = nn.Sequential(
            CNNBlock(in_c, mid_c, kernel_size=1),
            CNNBlock(mid_c, mid_c * 2, kernel_size=3, padding=1),
            CNNBlock(mid_c * 2, mid_c, kernel_size=1),
            CNNBlock(mid_c, mid_c * 2, kernel_size=3, padding=1),
            CNNBlock(mid_c * 2, mid_c, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class YOLOv3ResNet50(nn.Module):
    def __init__(self, C=20, anchors=DEFAULT_ANCHORS, pretrained=True):
        super().__init__()
        self.C = C
        self.B = len(anchors[0])
        assert len(anchors) == 3, "expected 3 scales"
        assert all(len(a) == self.B for a in anchors), "expected same B per scale"
        flat = [pair for scale in anchors for pair in scale]
        # [3 scales, B, 2] in normalized [0, 1] coords.
        self.register_buffer(
            "anchors", torch.tensor(flat, dtype=torch.float32).view(3, self.B, 2)
        )

        backbone = resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2  #  512ch, stride 8
        self.layer3 = backbone.layer3  # 1024ch, stride 16
        self.layer4 = backbone.layer4  # 2048ch, stride 32

        out_c = self.B * (5 + C)

        # Head widths match the Darknet-53 v3 head (paper fig. 3, table 1).
        # Coarsest scale (stride 32, 2048ch in).
        self.fpn_l = FPNBlock(2048, 512)
        self.head_l = nn.Sequential(
            CNNBlock(512, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, out_c, kernel_size=1),
        )
        # Top-down s32 -> s16: 1x1 reduce + 2x upsample, concat with layer3.
        self.up_l_to_m = CNNBlock(512, 256, kernel_size=1)
        self.fpn_m = FPNBlock(1024 + 256, 256)
        self.head_m = nn.Sequential(
            CNNBlock(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, out_c, kernel_size=1),
        )
        # Top-down s16 -> s8: 1x1 reduce + 2x upsample, concat with layer2.
        self.up_m_to_s = CNNBlock(256, 128, kernel_size=1)
        self.fpn_s = FPNBlock(512 + 128, 128)
        self.head_s = nn.Sequential(
            CNNBlock(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, out_c, kernel_size=1),
        )

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def backbone_parameters(self):
        for m in (self.stem, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def head_parameters(self):
        for m in (self.fpn_l, self.head_l, self.up_l_to_m,
                  self.fpn_m, self.head_m, self.up_m_to_s,
                  self.fpn_s, self.head_s):
            yield from m.parameters()

    def _reshape(self, p):
        N, _, S, _ = p.shape
        p = p.view(N, self.B, 5 + self.C, S, S)
        return p.permute(0, 3, 4, 1, 2).contiguous()

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.stem(x)
        f2 = self.layer2(x)   # [N, 128, H/8,  W/8 ]
        f3 = self.layer3(f2)  # [N, 256, H/16, W/16]
        f4 = self.layer4(f3)  # [N, 512, H/32, W/32]

        feat_l = self.fpn_l(f4)
        pred_l = self.head_l(feat_l)

        up = F.interpolate(self.up_l_to_m(feat_l), scale_factor=2, mode="nearest")
        feat_m = self.fpn_m(torch.cat([up, f3], dim=1))
        pred_m = self.head_m(feat_m)

        up = F.interpolate(self.up_m_to_s(feat_m), scale_factor=2, mode="nearest")
        feat_s = self.fpn_s(torch.cat([up, f2], dim=1))
        pred_s = self.head_s(feat_s)

        # Order matches anchors[0..2]: stride 8, 16, 32.
        return [self._reshape(pred_s), self._reshape(pred_m), self._reshape(pred_l)]


if __name__ == "__main__":
    m = YOLOv3ResNet50(pretrained=False)
    for size in (320, 416, 608):
        outs = m(torch.randn(2, 3, size, size))
        shapes = [tuple(o.shape) for o in outs]
        print(f"input {size} -> {shapes}")
