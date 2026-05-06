"""Evaluate a trained YOLO v1 checkpoint: mAP@0.5 and per-class AP."""
import argparse
import os
import torch
from torch.utils.data import DataLoader

from model import YOLOv1
from dataset import VOCDetection, VOC_CLASSES
from utils import get_bboxes, mean_average_precision


def per_class_ap(pred, true, num_classes=20):
    aps = {}
    for c in range(num_classes):
        p = [d for d in pred if d[1] == c]
        t = [g for g in true if g[1] == c]
        if not t:
            continue
        aps[VOC_CLASSES[c]] = mean_average_precision(p, t, num_classes=num_classes)
    return aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/imerit/data/pascal_voc_dataset/VOC_Detection")
    ap.add_argument("--ckpt", default="yolov1.pt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = VOCDetection(os.path.join(args.data_root, "test"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = YOLOv1().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    pred, true = get_bboxes(loader, model, iou_thr=args.iou, conf_thr=args.conf, device=device)
    mAP = mean_average_precision(pred, true, iou_thr=args.iou)
    print(f"mAP@{args.iou:.2f} = {mAP:.4f}")
    print("\nPer-class AP:")
    for name, v in sorted(per_class_ap(pred, true).items(), key=lambda x: -x[1]):
        print(f"  {name:14s} {v:.4f}")


if __name__ == "__main__":
    main()
