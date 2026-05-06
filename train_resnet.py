"""Train YOLO v1 with an ImageNet-pretrained ResNet-18 backbone on Pascal VOC.

Pretrained features need a gentler schedule than from-scratch training:
- Lower peak LR (1e-3) to avoid wrecking ImageNet features.
- Lower LR multiplier on the backbone (0.1x) vs the freshly-init detection head.
- Cosine anneal after a short warmup; ~135 epochs is plenty for fine-tuning.
"""
import argparse
import math
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from model_resnet import YOLOv1ResNet18
from dataset import VOCDetection
from loss import YoloLoss
from utils import get_bboxes, mean_average_precision


PEAK_LR = 1e-3
WARMUP_EPOCHS = 5
TOTAL_EPOCHS_DEFAULT = 135
FLOOR_LR = PEAK_LR / 100   # 1e-5
BACKBONE_LR_MULT = 0.1     # backbone trains 10x slower than the head


def lr_lambda_factory(total_epochs):
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (FLOOR_LR + (PEAK_LR - FLOOR_LR) * ((epoch + 1) / WARMUP_EPOCHS))
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
    ap.add_argument("--ckpt", default="yolov1_resnet18.pt")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip ImageNet weights (for sanity testing)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = VOCDetection(os.path.join(args.data_root, "train"), augment=True)
    test_ds = VOCDetection(os.path.join(args.data_root, "test"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = YOLOv1ResNet18(pretrained=not args.no_pretrained).to(device)

    # Two param groups: backbone gets a smaller base LR via its multiplier.
    optim = SGD(
        [
            {"params": list(model.backbone_parameters()), "lr": BACKBONE_LR_MULT},
            {"params": list(model.head_parameters()), "lr": 1.0},
        ],
        momentum=0.9, weight_decay=5e-4,
    )
    sched = LambdaLR(optim, lr_lambda=lr_lambda_factory(args.epochs))
    loss_fn = YoloLoss()

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
        print(f"resumed from {args.ckpt} at epoch {start_epoch} | prev mAP={prev_map:.4f}", flush=True)

    print(f"device={device} | train={len(train_ds)} | test={len(test_ds)} | "
          f"batches/epoch={len(train_loader)} | pretrained={not args.no_pretrained}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optim.step()
            running += loss.item()
            if i % 20 == 0:
                lrs = [pg["lr"] for pg in optim.param_groups]
                print(f"  epoch {epoch:03d} step {i:04d}/{len(train_loader)} | "
                      f"loss {loss.item():.4f} | lr_bb {lrs[0]:.2e} lr_hd {lrs[1]:.2e}", flush=True)
        sched.step()
        avg = running / len(train_loader)
        print(f"epoch {epoch:03d} | lr_hd {optim.param_groups[1]['lr']:.4g} | loss {avg:.4f}", flush=True)

        if (epoch + 1) % args.eval_every == 0:
            pred, true = get_bboxes(test_loader, model, device=device)
            mAP = mean_average_precision(pred, true)
            print(f"  val mAP@0.5 = {mAP:.4f}", flush=True)
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "mAP": mAP,
            }, args.ckpt)


if __name__ == "__main__":
    main()
