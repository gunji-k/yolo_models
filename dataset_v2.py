"""Pascal VOC dataset for YOLO v2 training.

Returns (img, padded_boxes). `padded_boxes` is a fixed-size [max_objs, 5]
tensor with rows (cls, cx, cy, w, h) in normalized [0, 1] coords. Empty rows
are padded with cls = -1. Anchor assignment is left to the loss so that
multi-scale training can vary `S` per batch.
"""
import os
import csv
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from dataset import VOC_CLASSES, CLASS_TO_IDX


class VOCDetectionV2(Dataset):
    def __init__(self, root, img_size=416, max_objs=50, augment=False):
        self.img_dir = os.path.join(root, "images")
        self.tgt_dir = os.path.join(root, "targets")
        self.ids = sorted(os.path.splitext(f)[0] for f in os.listdir(self.img_dir))
        self.img_size = img_size
        self.max_objs = max_objs
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _to_tensor(self, img):
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_boxes(self, stem, w, h):
        boxes = []
        with open(os.path.join(self.tgt_dir, stem + ".csv")) as f:
            for row in csv.DictReader(f):
                cls = CLASS_TO_IDX.get(row["object"])
                if cls is None:
                    continue
                xmin, ymin = float(row["xmin"]), float(row["ymin"])
                xmax, ymax = float(row["xmax"]), float(row["ymax"])
                cx = (xmin + xmax) / 2 / w
                cy = (ymin + ymax) / 2 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                boxes.append([cls, cx, cy, bw, bh])
        return boxes

    def _augment(self, img, boxes):
        # Horizontal flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[c, 1.0 - cx, cy, bw, bh] for c, cx, cy, bw, bh in boxes]

        # Random scale + translate (paper: up to 20% of image size)
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            max_dx = 0.2 * (1.0 / scale)
            max_dy = 0.2 * (1.0 / scale)
            dx = random.uniform(-max_dx, max_dx)
            dy = random.uniform(-max_dy, max_dy)
            new_boxes = []
            for c, cx, cy, bw, bh in boxes:
                cx = (cx + dx) * scale
                cy = (cy + dy) * scale
                bw = bw * scale
                bh = bh * scale
                if 0 < cx < 1 and 0 < cy < 1:
                    bw = min(bw, 2 * cx, 2 * (1 - cx))
                    bh = min(bh, 2 * cy, 2 * (1 - cy))
                    if bw > 0.01 and bh > 0.01:
                        new_boxes.append([c, cx, cy, bw, bh])
            if new_boxes:
                boxes = new_boxes
                w, h = img.size
                crop_x1 = max(0, min(int(-dx * w * scale), w - 1))
                crop_y1 = max(0, min(int(-dy * h * scale), h - 1))
                crop_x2 = min(w, crop_x1 + int(w / scale))
                crop_y2 = min(h, crop_y1 + int(h / scale))
                img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Color jitter
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))

        return img, boxes

    def __getitem__(self, idx):
        stem = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, stem + ".jpg")).convert("RGB")
        w, h = img.size
        boxes = self._load_boxes(stem, w, h)

        if self.augment:
            img, boxes = self._augment(img, boxes)

        img = self._to_tensor(img)

        # Padded box tensor; cls=-1 is the padding marker.
        target = torch.full((self.max_objs, 5), -1.0)
        for i, b in enumerate(boxes[: self.max_objs]):
            target[i] = torch.tensor(b)
        return img, target


__all__ = ["VOCDetectionV2", "VOC_CLASSES", "CLASS_TO_IDX"]
