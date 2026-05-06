"""YOLO v2 with an ImageNet-pretrained ResNet-18 backbone (arXiv 1612.08242).

Differences vs the v1 model:
- Fully convolutional head (no FC, no dropout) -> final 1x1 conv to B*(5+C).
- 416 input, 13x13 grid (stride 32), B=5 anchor priors per cell.
- Per-anchor class predictions; output reshaped to [N, S, S, B, 5+C].
- Passthrough layer: space-to-depth from layer3 (26x26x256) concat into the 13x13 head.
- Predictions are raw t_x, t_y, t_w, t_h, t_o, class logits. The loss/decoder
  apply sigmoid(tx,ty,to), exp(tw,th)*prior, and softmax(class).

Multi-scale training is supported: any input HxW that's a multiple of 32 works.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Default VOC anchors from the YOLOv2 paper / yolov2-voc.cfg, in 13x13 grid units.
# Replace these with k-means anchors fit on your training set for best results.
DEFAULT_VOC_ANCHORS = (
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
)


class CNNBlock(nn.Module):
    def __init__(self, in_c, out_c, **kw):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, bias=False, **kw)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Reorg(nn.Module):
    """Space-to-depth: (N, C, H, W) -> (N, C*s*s, H/s, W/s)."""

    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        s = self.stride
        N, C, H, W = x.shape
        assert H % s == 0 and W % s == 0, "feature map must be divisible by stride"
        x = x.view(N, C, H // s, s, W // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(N, C * s * s, H // s, W // s)


class YOLOv2ResNet18(nn.Module):
    def __init__(self, C=20, anchors=DEFAULT_VOC_ANCHORS, pretrained=True):
        super().__init__()
        self.C = C
        self.B = len(anchors)
        # Anchors are kept in grid units (cells), so they're scale-invariant under
        # multi-scale training - the loss/decoder just multiplies by S at use time.
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))

        backbone = resnet18(pretrained=pretrained)
        # Split the backbone so we can tap layer3 (stride 16) for the passthrough.
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        self.layer3 = backbone.layer3   # 256ch, stride 16 (26x26 at 416 input)
        self.layer4 = backbone.layer4   # 512ch, stride 32 (13x13 at 416 input)

        # Detection neck on the 13x13 features.
        self.neck = nn.Sequential(
            CNNBlock(512, 1024, kernel_size=3, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, padding=1),
        )

        # Passthrough route: 1x1 bottleneck on the 26x26 features, then reorg to 13x13.
        self.route = CNNBlock(256, 64, kernel_size=1)
        self.reorg = Reorg(stride=2)  # 64x26x26 -> 256x13x13

        # Merge passthrough with main branch and predict.
        self.post = CNNBlock(1024 + 256, 1024, kernel_size=3, padding=1)
        self.predict = nn.Conv2d(1024, self.B * (5 + C), kernel_size=1)

        # ImageNet normalization baked in (input is /255 from the dataset).
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def backbone_parameters(self):
        for m in (self.stem, self.layer3, self.layer4):
            yield from m.parameters()

    def head_parameters(self):
        for m in (self.neck, self.route, self.post, self.predict):
            yield from m.parameters()

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.stem(x)
        f3 = self.layer3(x)            # [N, 256, H/16, W/16]
        f4 = self.layer4(f3)           # [N, 512, H/32, W/32]

        y = self.neck(f4)              # [N, 1024, S, S]
        p = self.reorg(self.route(f3)) # [N, 256, S, S]
        y = torch.cat([y, p], dim=1)   # [N, 1280, S, S]
        y = self.post(y)               # [N, 1024, S, S]
        y = self.predict(y)            # [N, B*(5+C), S, S]

        # Reshape to [N, S, S, B, 5+C].
        N, _, S, _ = y.shape
        y = y.view(N, self.B, 5 + self.C, S, S)
        return y.permute(0, 3, 4, 1, 2).contiguous()


if __name__ == "__main__":
    m = YOLOv2ResNet18(pretrained=False)
    for size in (320, 416, 608):
        y = m(torch.randn(2, 3, size, size))
        print(f"input {size} -> output {tuple(y.shape)}")
    # input 320 -> (2, 10, 10, 5, 25)
    # input 416 -> (2, 13, 13, 5, 25)
    # input 608 -> (2, 19, 19, 5, 25)
