"""YOLO v1 with an ImageNet-pretrained ResNet-18 backbone.

ResNet-18 stride-32 → 448 input gives a 14x14 feature map. A stride-2 detection
conv brings it to 7x7, then a fully-convolutional head predicts S*S*(C+B*5).
ImageNet normalization is applied inside the model so the existing dataset
(/255 only) can be reused unchanged.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class CNNBlock(nn.Module):
    def __init__(self, in_c, out_c, **kw):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, bias=False, **kw)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOv1ResNet18(nn.Module):
    def __init__(self, S=7, B=2, C=20, pretrained=True):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        backbone = resnet18(pretrained=pretrained)
        # Drop avgpool + fc; keep the conv stem through layer4 (output: 512x14x14 for 448 input)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Detection neck: 14x14x512 -> 7x7x1024 -> 7x7x1024
        self.neck = nn.Sequential(
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        )

        # FC head matches the original YOLOv1 head shape
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, S * S * (C + B * 5)),
        )

        # ImageNet normalization baked in (input is /255 from the dataset)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        for m in (self.neck, self.head):
            for p in m.parameters():
                yield p

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x.view(-1, self.S, self.S, self.C + self.B * 5)


if __name__ == "__main__":
    m = YOLOv1ResNet18(pretrained=False)
    y = m(torch.randn(2, 3, 448, 448))
    print(y.shape)  # [2, 7, 7, 30]
