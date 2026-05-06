"""Train YOLO v1 on Pascal VOC detection."""
import argparse
import math
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from model import YOLOv1
from dataset import VOCDetection
from loss import YoloLoss
from utils import get_bboxes, mean_average_precision


PEAK_LR = 2.5e-3  # paper uses 1e-2 at bs=64; linear-scaled for bs=16

# Extended-training phase: warm-restart from the converged epoch-334 weights,
# then cosine-anneal over the remaining epochs.
RESTART_EPOCH = 335
RESTART_WARMUP = 5
RESTART_PEAK = PEAK_LR / 10      # 2.5e-4 — highest LR the prior schedule reused mid-training
RESTART_FLOOR = PEAK_LR / 1000   # 2.5e-6 — matches prior schedule's terminal LR
RESTART_TOTAL = 200              # new epochs added (335..534)


def lr_schedule(epoch):
    if epoch < 5:
        return 1e-4 + (PEAK_LR - 1e-4) * (epoch / 5)
    if epoch < 80:
        return PEAK_LR
    if epoch < 110:
        return PEAK_LR / 10
    if epoch < 135:
        return PEAK_LR / 100
    if epoch < 305:
        return PEAK_LR / 10
    if epoch < 325:
        return PEAK_LR / 100
    if epoch < RESTART_EPOCH:
        return PEAK_LR / 1000
    e = epoch - RESTART_EPOCH
    if e < RESTART_WARMUP:
        return RESTART_FLOOR + (RESTART_PEAK - RESTART_FLOOR) * ((e + 1) / RESTART_WARMUP)
    if e >= RESTART_TOTAL:
        return RESTART_FLOOR
    t = (e - RESTART_WARMUP) / (RESTART_TOTAL - RESTART_WARMUP)
    return RESTART_FLOOR + 0.5 * (RESTART_PEAK - RESTART_FLOOR) * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/imerit/data/pascal_voc_dataset/VOC_Detection")
    ap.add_argument("--epochs", type=int, default=535)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", default="yolov1.pt")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = VOCDetection(os.path.join(args.data_root, "train"), augment=True)
    test_ds = VOCDetection(os.path.join(args.data_root, "test"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = YOLOv1().to(device)
    optim = SGD(model.parameters(), lr=1.0, momentum=0.9, weight_decay=5e-4)
    sched = LambdaLR(optim, lr_lambda=lr_schedule)
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
        for pg, base_lr in zip(optim.param_groups, sched.base_lrs):
            pg["lr"] = base_lr * lr_schedule(sched.last_epoch)
        prev_map = ckpt.get("mAP", float("nan"))
        del ckpt
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"resumed from {args.ckpt} at epoch {start_epoch} | prev mAP={prev_map:.4f}", flush=True)

    print(f"device={device} | train={len(train_ds)} | test={len(test_ds)} | batches/epoch={len(train_loader)}", flush=True)

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
                print(f"  epoch {epoch:03d} step {i:04d}/{len(train_loader)} | loss {loss.item():.4f}", flush=True)
        sched.step()
        avg = running / len(train_loader)
        print(f"epoch {epoch:03d} | lr {optim.param_groups[0]['lr']:.4g} | loss {avg:.4f}", flush=True)

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
