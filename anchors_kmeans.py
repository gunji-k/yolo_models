"""Fit YOLOv2 anchor priors via k-means with IoU distance.

Uses the training set's box widths/heights (normalized to image size). Output
anchors are printed in two forms:
  - normalized (w, h) in [0, 1]
  - grid units for S=13 (i.e. multiplied by 13) -> ready for yolov2_resnet18.py
"""
import argparse
import os
import csv
import numpy as np


def load_wh(targets_dir, images_dir):
    """Return (w, h) box dimensions normalized by image size for every box."""
    from PIL import Image  # local import: only needed when running this script

    whs = []
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir))
    for stem in stems:
        img_path = os.path.join(images_dir, stem + ".jpg")
        with Image.open(img_path) as im:
            iw, ih = im.size
        with open(os.path.join(targets_dir, stem + ".csv")) as f:
            for row in csv.DictReader(f):
                xmin, ymin = float(row["xmin"]), float(row["ymin"])
                xmax, ymax = float(row["xmax"]), float(row["ymax"])
                whs.append(((xmax - xmin) / iw, (ymax - ymin) / ih))
    return np.asarray(whs, dtype=np.float64)


def wh_iou(boxes, anchors):
    """IoU of (w, h) pairs centered at the origin.
    boxes: [N, 2], anchors: [K, 2] -> [N, K]."""
    inter_w = np.minimum(boxes[:, None, 0], anchors[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 1], anchors[None, :, 1])
    inter = inter_w * inter_h
    a_box = (boxes[:, 0] * boxes[:, 1])[:, None]
    a_anc = (anchors[:, 0] * anchors[:, 1])[None, :]
    return inter / (a_box + a_anc - inter + 1e-9)


def kmeans_iou(boxes, k=5, max_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    centroids = boxes[rng.choice(len(boxes), k, replace=False)].copy()
    last_assign = None
    for _ in range(max_iter):
        ious = wh_iou(boxes, centroids)
        assign = ious.argmax(axis=1)
        if last_assign is not None and np.array_equal(assign, last_assign):
            break
        last_assign = assign
        for c in range(k):
            members = boxes[assign == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    mean_iou = wh_iou(boxes, centroids).max(axis=1).mean()
    # Sort by area for nicer printing.
    centroids = centroids[np.argsort(centroids[:, 0] * centroids[:, 1])]
    return centroids, mean_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/imerit/data/pascal_voc_dataset/VOC_Detection")
    ap.add_argument("--split", default="train")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--grid-size", type=int, default=13,
                    help="S used for the detection head (13 for 416 input)")
    ap.add_argument("--out", default="",
                    help="if set, write normalized anchors to this file "
                         "(one 'w,h' per line) for consumption by train_v3.py")
    args = ap.parse_args()

    images = os.path.join(args.data_root, args.split, "images")
    targets = os.path.join(args.data_root, args.split, "targets")
    boxes = load_wh(targets, images)
    print(f"loaded {len(boxes)} boxes from {args.split}")
    anchors, mean_iou = kmeans_iou(boxes, k=args.k)
    print(f"\nmean IoU vs nearest centroid: {mean_iou:.4f}\n")
    print("normalized (w, h):")
    for w, h in anchors:
        print(f"  ({w:.5f}, {h:.5f})")
    print(f"\ngrid units (S={args.grid_size}):")
    for w, h in anchors * args.grid_size:
        print(f"  ({w:.5f}, {h:.5f})")

    if args.out:
        with open(args.out, "w") as f:
            for w, h in anchors:
                f.write(f"{w:.6f},{h:.6f}\n")
        print(f"\nwrote {len(anchors)} normalized anchors to {args.out}")


if __name__ == "__main__":
    main()
