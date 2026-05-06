"""Train YOLO v2 with an ImageNet-pretrained ResNet-18 backbone on Pascal VOC.

- Multi-scale training: every `--multiscale-period` iterations sample a new
  size from {320, 352, ..., 608} and resize the batch (boxes are normalized so
  no rescaling needed).
- Same two-LR-group schedule as train_resnet.py (backbone slower than head).
"""
import argparse
import math
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from yolov2_resnet18 import YOLOv2ResNet18, DEFAULT_VOC_ANCHORS
from dataset_v2 import VOCDetectionV2
from loss_v2 import YoloV2Loss
from utils_v2 import get_bboxes_v2
from utils import mean_average_precision


PEAK_LR = 1e-3
WARMUP_EPOCHS = 5
TOTAL_EPOCHS_DEFAULT = 160
FLOOR_LR = PEAK_LR / 100
BACKBONE_LR_MULT = 0.1
MULTISCALE_SIZES = list(range(320, 608 + 1, 32))  # 320, 352, ..., 608


def lr_lambda_factory(total_epochs):
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return FLOOR_LR + (PEAK_LR - FLOOR_LR) * ((epoch + 1) / WARMUP_EPOCHS)
        if epoch >= total_epochs:
            return FLOOR_LR
        t = (epoch - WARMUP_EPOCHS) / max(1, total_epochs - WARMUP_EPOCHS)
        return FLOOR_LR + 0.5 * (PEAK_LR - FLOOR_LR) * (1 + math.cos(math.pi * t))
    return lr_lambda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/imerit/data/pascal_voc_dataset/VOC_Detection")
    ap.add_argument("--epochs", type=int, default=TOTAL_EPOCHS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", default="yolov2_resnet18.pt")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--anchors", default="",
                    help="comma-separated 'w,h;w,h;...' in grid units (S=13). "
                         "If empty, uses the YOLOv2 paper VOC anchors.")
    ap.add_argument("--base-size", type=int, default=608,
                    help="dataset image size; train batches are then downsampled "
                         "to a multi-scale value")
    ap.add_argument("--eval-size", type=int, default=416)
    ap.add_argument("--multiscale-period", type=int, default=10,
                    help="resample input size every N training iterations")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.anchors:
        anchors = [tuple(float(v) for v in pair.split(","))
                   for pair in args.anchors.split(";") if pair]
    else:
        anchors = list(DEFAULT_VOC_ANCHORS)

    train_ds = VOCDetectionV2(os.path.join(args.data_root, "train"),
                              img_size=args.base_size, augment=True)
    test_ds = VOCDetectionV2(os.path.join(args.data_root, "test"),
                             img_size=args.eval_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = YOLOv2ResNet18(C=20, anchors=anchors,
                           pretrained=not args.no_pretrained).to(device)
    optim = SGD(
        [
            {"params": list(model.backbone_parameters()), "lr": BACKBONE_LR_MULT},
            {"params": list(model.head_parameters()), "lr": 1.0},
        ],
        momentum=0.9, weight_decay=5e-4,
    )
    sched = LambdaLR(optim, lr_lambda=lr_lambda_factory(args.epochs))
    loss_fn = YoloV2Loss(anchors, num_classes=20).to(device)

    start_epoch = 0
    if args.resume and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", -1) + 1
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        else:
            for _ in range(start_epoch):
                sched.step()
        prev_map = ckpt.get("mAP", float("nan"))
        del ckpt
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"resumed from {args.ckpt} at epoch {start_epoch} | prev mAP={prev_map:.4f}",
              flush=True)

    print(f"device={device} | train={len(train_ds)} | test={len(test_ds)} | "
          f"batches/epoch={len(train_loader)} | anchors={len(anchors)}", flush=True)

    cur_size = args.base_size
    iteration = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for i, (imgs, target_boxes) in enumerate(train_loader):
            # Multi-scale: pick a new size periodically.
            if iteration % args.multiscale_period == 0:
                cur_size = random.choice(MULTISCALE_SIZES)
            imgs = imgs.to(device, non_blocking=True)
            target_boxes = target_boxes.to(device, non_blocking=True)
            if cur_size != imgs.shape[-1]:
                imgs = F.interpolate(imgs, size=cur_size, mode="bilinear", align_corners=False)

            preds = model(imgs)
            loss = loss_fn(preds, target_boxes)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optim.step()
            running += loss.item()
            iteration += 1

            if i % 20 == 0:
                lrs = [pg["lr"] for pg in optim.param_groups]
                print(f"  epoch {epoch:03d} step {i:04d}/{len(train_loader)} | "
                      f"size {cur_size} | loss {loss.item():.4f} | "
                      f"lr_bb {lrs[0]:.2e} lr_hd {lrs[1]:.2e}", flush=True)
        sched.step()
        avg = running / len(train_loader)
        print(f"epoch {epoch:03d} | lr_hd {optim.param_groups[1]['lr']:.4g} | "
              f"loss {avg:.4f}", flush=True)

        if (epoch + 1) % args.eval_every == 0:
            pred, true = get_bboxes_v2(test_loader, model, anchors, device=device)
            mAP = mean_average_precision(pred, true)
            print(f"  val mAP@0.5 = {mAP:.4f}", flush=True)
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "mAP": mAP,
                "anchors": anchors,
            }, args.ckpt)


if __name__ == "__main__":
    main()
