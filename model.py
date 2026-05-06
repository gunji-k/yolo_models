"""YOLO v1 architecture (Redmon et al., 2016).

24 convolutional layers + 2 fully connected layers. Input 448x448x3,
output S*S*(B*5 + C) = 7*7*30 for Pascal VOC.
"""
import torch
import torch.nn as nn

# (kernel, out_channels, stride, padding) or "M" for maxpool 2x2 s2
# List-of-tuples => repeat block
_CFG = [
    (7, 64, 2, 3), "M",
    (3, 192, 1, 1), "M",
    (1, 128, 1, 0), (3, 256, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], (1, 512, 1, 0), (3, 1024, 1, 1), "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1), (3, 1024, 2, 1),
    (3, 1024, 1, 1), (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_c, out_c, **kw):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, bias=False, **kw)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.darknet = self._build(_CFG, in_c=3)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, S * S * (C + B * 5)),
        )

    @staticmethod
    def _build(cfg, in_c):
        layers = []
        for x in cfg:
            if x == "M":
                layers.append(nn.MaxPool2d(2, 2))
            elif isinstance(x, tuple):
                k, oc, s, p = x
                layers.append(CNNBlock(in_c, oc, kernel_size=k, stride=s, padding=p))
                in_c = oc
            elif isinstance(x, list):
                a, b, n = x
                for _ in range(n):
                    layers.append(CNNBlock(in_c, a[1], kernel_size=a[0], stride=a[2], padding=a[3]))
                    layers.append(CNNBlock(a[1], b[1], kernel_size=b[0], stride=b[2], padding=b[3]))
                    in_c = b[1]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.darknet(x)
        x = self.head(x)
        return x.view(-1, self.S, self.S, self.C + self.B * 5)


if __name__ == "__main__":
    m = YOLOv1()
    y = m(torch.randn(2, 3, 448, 448))
    print(y.shape)  # [2, 7, 7, 30]
