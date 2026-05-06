"""Visualize YOLO v1 detections on images."""
import argparse
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from model import YOLOv1
from dataset import VOC_CLASSES
from utils import cells_to_boxes, nms


def detect(model, img_path, device, conf_thr=0.4, iou_thr=0.5):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    resized = img.resize((448, 448), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
    # Diagnostic: raw objectness stats across the 7x7x2 predictors
    C, B = 20, 2
    obj_slots = torch.cat([pred[..., C + b * 5:C + 1 + b * 5] for b in range(B)], dim=-1)
    print(f"  raw obj: min={obj_slots.min().item():.3f} "
          f"max={obj_slots.max().item():.3f} "
          f"mean={obj_slots.mean().item():.3f}")
    boxes_all = cells_to_boxes(pred)[0].tolist()
    top = sorted(boxes_all, key=lambda b: b[1], reverse=True)[:3]
    print(f"  top-3 scores (after clamp): {[round(b[1], 4) for b in top]}")
    return img, nms(boxes_all, iou_thr, conf_thr), w, h


def draw(img, boxes, w, h, out_path):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    cmap = plt.get_cmap("tab20")
    for cls, conf, cx, cy, bw, bh in boxes:
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        rect = patches.Rectangle((x1, y1), bw * w, bh * h, linewidth=2,
                                 edgecolor=cmap(int(cls) % 20), facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f"{VOC_CLASSES[int(cls)]} {conf:.2f}",
                color="white", fontsize=10,
                bbox=dict(facecolor=cmap(int(cls) % 20), edgecolor="none", pad=1))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="yolov1.pt")
    ap.add_argument("--images", nargs="*", default=[], help="explicit image paths")
    ap.add_argument("--random", type=int, default=0,
                    help="sample N random images from --test-dir instead of --images")
    ap.add_argument("--test-dir",
                    default="/home/imerit/data/pascal_voc_dataset/VOC_Detection/test/images")
    ap.add_argument("--out-dir", default="vis_out")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"
    model = YOLOv1().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    paths = list(args.images)
    if args.random > 0:
        random.seed(args.seed)
        pool = [os.path.join(args.test_dir, f)
                for f in os.listdir(args.test_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        paths += random.sample(pool, min(args.random, len(pool)))
    if not paths:
        raise SystemExit("no images: pass --images or --random N")

    for p in paths:
        img, boxes, w, h = detect(model, p, device, args.conf, args.iou)
        print(f"{os.path.basename(p)}: {len(boxes)} detections")
        draw(img, boxes, w, h, os.path.join(args.out_dir, os.path.basename(p)))


if __name__ == "__main__":
    main()
