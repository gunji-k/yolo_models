"""Train YOLOv3 with ImageNet-pretrained ResNet-18 backbone on Pascal VOC.

Same multi-scale training and 2-LR-group schedule as train_v2.py. Reuses
VOCDetectionV2 since the dataset format (padded raw GT boxes) is identical.

Anchors: 9 priors as 9 (w, h) pairs in normalized [0, 1]. Pass via --anchors;
anchors are auto-sorted by area and split 3-per-scale (small -> stride 8).
"""
import argparse
import math
import os
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from model_v3 import YOLOv3ResNet50, DEFAULT_ANCHORS
from dataset_v2 import VOCDetectionV2
from loss_v3 import YoloV3Loss
from utils_v3 import get_bboxes_v3
from utils import mean_average_precision


PEAK_LR = 1e-3
WARMUP_EPOCHS = 5
TOTAL_EPOCHS_DEFAULT = 160
FLOOR_LR = PEAK_LR / 100
BACKBONE_LR_MULT = 0.1
MULTISCALE_SIZES = list(range(320, 608 + 1, 32))


class ModelEMA:
    """Polyak/EMA of model weights and floating-point buffers (incl. BN
    running_mean/running_var). Decay ramps as `decay·(1 − exp(−n/tau))` so the
    EMA tracks the live model closely for the first few thousand updates and
    then stabilizes — same schedule as YOLOv5/timm.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = updates
        self.decay_fn = lambda n: decay * (1 - math.exp(-n / tau))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay_fn(self.updates)
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def load_anchors(path):
    """Read 9 normalized 'w,h' pairs (one per line) from `path`. Sort by area
    and split 3-per-scale (smallest -> stride 8 head)."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            pairs.append((float(parts[0]), float(parts[1])))
    assert len(pairs) == 9, f"expected 9 anchors in {path}, got {len(pairs)}"
    pairs = sorted(pairs, key=lambda wh: wh[0] * wh[1])
    return [list(pairs[0:3]), list(pairs[3:6]), list(pairs[6:9])]


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
    ap.add_argument("--ckpt", default="yolov3_resnet50.pt")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--anchors", default="",
                    help="path to a file with 9 normalized 'w,h' anchors, one "
                         "per line (output of `anchors_kmeans.py -k 9 --out`). "
                         "If empty, uses DEFAULT_ANCHORS.")
    ap.add_argument("--base-size", type=int, default=608,
                    help="dataset image size; train batches are downsampled "
                         "to a multi-scale value")
    ap.add_argument("--eval-size", type=int, default=416)
    ap.add_argument("--multiscale-period", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.anchors:
        anchors = load_anchors(args.anchors)
    else:
        anchors = DEFAULT_ANCHORS

    train_ds = VOCDetectionV2(os.path.join(args.data_root, "train"),
                              img_size=args.base_size, augment=True)
    test_ds = VOCDetectionV2(os.path.join(args.data_root, "test"),
                             img_size=args.eval_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = YOLOv3ResNet50(C=20, anchors=anchors,
                           pretrained=not args.no_pretrained).to(device)
    optim = SGD(
        [
            {"params": list(model.backbone_parameters()), "lr": BACKBONE_LR_MULT},
            {"params": list(model.head_parameters()), "lr": 1.0},
        ],
        momentum=0.9, weight_decay=5e-4,
    )
    sched = LambdaLR(optim, lr_lambda=lr_lambda_factory(args.epochs))
    loss_fn = YoloV3Loss(anchors, num_classes=20).to(device)
    ema = ModelEMA(model)

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
        if "ema" in ckpt:
            ema.module.load_state_dict(ckpt["ema"])
            ema.updates = ckpt.get("ema_updates", 0)
        else:
            ema = ModelEMA(model)
        prev_map = ckpt.get("mAP", float("nan"))
        del ckpt
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"resumed from {args.ckpt} at epoch {start_epoch} | prev mAP={prev_map:.4f} | "
              f"ema updates={ema.updates}", flush=True)

    print(f"device={device} | train={len(train_ds)} | test={len(test_ds)} | "
          f"batches/epoch={len(train_loader)} | anchors=9 (3 per scale)", flush=True)

    cur_size = args.base_size
    iteration = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        running_components = {"xy": 0.0, "wh": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
        for i, (imgs, target_boxes) in enumerate(train_loader):
            if iteration % args.multiscale_period == 0:
                cur_size = random.choice(MULTISCALE_SIZES)
            imgs = imgs.to(device, non_blocking=True)
            target_boxes = target_boxes.to(device, non_blocking=True)
            if cur_size != imgs.shape[-1]:
                imgs = F.interpolate(imgs, size=cur_size, mode="bilinear", align_corners=False)

            preds_list = model(imgs)
            loss, components = loss_fn(preds_list, target_boxes)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optim.step()
            ema.update(model)
            running += loss.item()
            for k in running_components:
                running_components[k] += components[k].item()
            iteration += 1

            if i % 20 == 0:
                lrs = [pg["lr"] for pg in optim.param_groups]
                comp_str = " ".join(f"{k} {components[k].item():.2f}" for k in
                                    ("xy", "wh", "obj", "noobj", "cls"))
                print(f"  epoch {epoch:03d} step {i:04d}/{len(train_loader)} | "
                      f"size {cur_size} | loss {loss.item():.4f} | {comp_str} | "
                      f"lr_bb {lrs[0]:.2e} lr_hd {lrs[1]:.2e}", flush=True)
        sched.step()
        avg = running / len(train_loader)
        avg_comp = {k: v / len(train_loader) for k, v in running_components.items()}
        avg_comp_str = " ".join(f"{k} {avg_comp[k]:.2f}" for k in
                                ("xy", "wh", "obj", "noobj", "cls"))
        print(f"epoch {epoch:03d} | lr_hd {optim.param_groups[1]['lr']:.4g} | "
              f"loss {avg:.4f} | {avg_comp_str}", flush=True)

        if (epoch + 1) % args.eval_every == 0:
            pred, true = get_bboxes_v3(test_loader, ema.module, anchors, device=device)
            mAP = mean_average_precision(pred, true)
            pred_live, true_live = get_bboxes_v3(test_loader, model, anchors, device=device)
            mAP_live = mean_average_precision(pred_live, true_live)
            print(f"  test mAP@0.5 (ema)  = {mAP:.4f}", flush=True)
            print(f"  test mAP@0.5 (live) = {mAP_live:.4f}", flush=True)
            torch.save({
                "model": model.state_dict(),
                "ema": ema.module.state_dict(),
                "ema_updates": ema.updates,
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "mAP": mAP,
                "mAP_live": mAP_live,
                "anchors": anchors,
            }, args.ckpt)


if __name__ == "__main__":
    main()
