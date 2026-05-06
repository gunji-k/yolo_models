"""Pascal VOC detection dataset loader for YOLO v1.

Expects the following layout (matches /home/imerit/data/pascal_voc_dataset/VOC_Detection):
    root/
        images/{id}.jpg
        targets/{id}.csv   # header: object,xmin,ymin,xmax,ymax  (absolute pixel coords)
"""
import os
import csv
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


class VOCDetection(Dataset):
    def __init__(self, root, S=7, B=2, C=20, img_size=448, augment=False):
        self.img_dir = os.path.join(root, "images")
        self.tgt_dir = os.path.join(root, "targets")
        self.ids = sorted(os.path.splitext(f)[0] for f in os.listdir(self.img_dir))
        self.S, self.B, self.C, self.img_size = S, B, C, img_size
        self.augment = augment

    def _to_tensor(self, img):
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __len__(self):
        return len(self.ids)

    def _load_boxes(self, stem, w, h):
        boxes = []
        path = os.path.join(self.tgt_dir, stem + ".csv")
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
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
        # --- Horizontal flip (50%) ---
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[cls, 1.0 - cx, cy, bw, bh]
                     for cls, cx, cy, bw, bh in boxes]

        # --- Random scale + translate (paper: up to 20% of image size) ---
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            max_dx = 0.2 * (1.0 / scale)
            max_dy = 0.2 * (1.0 / scale)
            dx = random.uniform(-max_dx, max_dx)
            dy = random.uniform(-max_dy, max_dy)
            new_boxes = []
            for cls, cx, cy, bw, bh in boxes:
                cx = (cx + dx) * scale
                cy = (cy + dy) * scale
                bw = bw * scale
                bh = bh * scale
                if 0 < cx < 1 and 0 < cy < 1:
                    bw = min(bw, 2 * cx, 2 * (1 - cx))
                    bh = min(bh, 2 * cy, 2 * (1 - cy))
                    if bw > 0.01 and bh > 0.01:
                        new_boxes.append([cls, cx, cy, bw, bh])
            if new_boxes:
                boxes = new_boxes
                w, h = img.size
                crop_x1 = int(-dx * w * scale)
                crop_y1 = int(-dy * h * scale)
                new_w = int(w / scale)
                new_h = int(h / scale)
                crop_x1 = max(0, min(crop_x1, w - 1))
                crop_y1 = max(0, min(crop_y1, h - 1))
                crop_x2 = min(w, crop_x1 + new_w)
                crop_y2 = min(h, crop_y1 + new_h)
                img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # --- Color jitter (paper: exposure and saturation up to 1.5x in HSV) ---
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            img = ImageEnhance.Color(img).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            img = ImageEnhance.Contrast(img).enhance(factor)

        return img, boxes

    def __getitem__(self, idx):
        stem = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, stem + ".jpg")).convert("RGB")
        w, h = img.size
        boxes = self._load_boxes(stem, w, h)

        if self.augment:
            img, boxes = self._augment(img, boxes)

        img = self._to_tensor(img)

        S, B, C = self.S, self.B, self.C
        target = torch.zeros((S, S, C + 5 * B))
        for cls, cx, cy, bw, bh in boxes:
            i, j = int(S * cy), int(S * cx)
            if i >= S or j >= S:
                continue
            if target[i, j, C] == 1:
                continue
            x_cell = S * cx - j
            y_cell = S * cy - i
            w_cell = bw * S
            h_cell = bh * S
            target[i, j, C] = 1
            target[i, j, C + 1:C + 5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
            target[i, j, cls] = 1
        return img, target
