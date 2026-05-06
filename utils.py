"""Geometry, NMS, and mAP utilities for YOLO v1."""
from collections import Counter
import torch


def iou_xywh(a, b):
    """IoU between boxes in (cx, cy, w, h) form. Broadcasts over leading dims."""
    ax1 = a[..., 0:1] - a[..., 2:3] / 2
    ay1 = a[..., 1:2] - a[..., 3:4] / 2
    ax2 = a[..., 0:1] + a[..., 2:3] / 2
    ay2 = a[..., 1:2] + a[..., 3:4] / 2
    bx1 = b[..., 0:1] - b[..., 2:3] / 2
    by1 = b[..., 1:2] - b[..., 3:4] / 2
    bx2 = b[..., 0:1] + b[..., 2:3] / 2
    by2 = b[..., 1:2] + b[..., 3:4] / 2

    inter_w = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(min=0)
    inter_h = (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def cells_to_boxes(preds, S=7, B=2, C=20):
    """Convert [N, S, S, C+5B] predictions to [N, S*S, 6] (class, conf, cx, cy, w, h),
    coords normalized to whole image."""
    N = preds.shape[0]
    preds = preds.detach().cpu()
    # Pick best of the B predicted boxes per cell (highest confidence)
    confs = torch.stack([preds[..., C + b * 5] for b in range(B)], dim=0)
    best_conf, best_idx = confs.max(dim=0)  # [N,S,S]
    boxes = torch.zeros(N, S, S, 4)
    for b in range(B):
        mask = (best_idx == b).unsqueeze(-1)
        boxes += mask * preds[..., C + 1 + b * 5 : C + 5 + b * 5]

    cell_indices = torch.arange(S).repeat(N, S, 1)  # [N,S,S] cols
    cx = (boxes[..., 0:1] + cell_indices.unsqueeze(-1)) / S
    cy = (boxes[..., 1:2] + cell_indices.permute(0, 2, 1).unsqueeze(-1)) / S
    w = boxes[..., 2:3] / S
    h = boxes[..., 3:4] / S

    # YOLOv1 regresses class scores with MSE against one-hot, so raw outputs are
    # already probability-like — do NOT softmax. Just clamp for safety.
    class_probs = preds[..., :C].clamp(0.0, 1.0)
    best_cls_prob, class_pred = class_probs.max(dim=-1, keepdim=True)
    class_pred = class_pred.float()
    obj = best_conf.unsqueeze(-1).clamp(0.0, 1.0)
    conf = obj * best_cls_prob
    out = torch.cat([class_pred, conf, cx, cy, w, h], dim=-1)  # [N,S,S,6]
    return out.reshape(N, S * S, 6)


def nms(boxes, iou_thr=0.5, conf_thr=0.4):
    """boxes: list of [class, conf, cx, cy, w, h]. Returns filtered list."""
    boxes = [b for b in boxes if b[1] > conf_thr]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    kept = []
    while boxes:
        top = boxes.pop(0)
        kept.append(top)
        boxes = [
            b for b in boxes
            if b[0] != top[0]
            or iou_xywh(torch.tensor(top[2:]), torch.tensor(b[2:])).item() < iou_thr
        ]
    return kept


def mean_average_precision(pred_boxes, true_boxes, iou_thr=0.5, num_classes=20):
    """VOC-style mAP at a single IoU threshold.

    pred_boxes/true_boxes: list of [img_idx, class, conf, cx, cy, w, h]
    """
    APs = []
    eps = 1e-6
    for c in range(num_classes):
        detections = [d for d in pred_boxes if d[1] == c]
        ground_truths = [g for g in true_boxes if g[1] == c]
        if not ground_truths:
            continue
        gt_per_img = Counter(g[0] for g in ground_truths)
        gt_per_img = {k: torch.zeros(v) for k, v in gt_per_img.items()}
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_gt = len(ground_truths)

        for i, det in enumerate(detections):
            gts = [g for g in ground_truths if g[0] == det[0]]
            best_iou, best_j = 0, -1
            for j, gt in enumerate(gts):
                iou = iou_xywh(torch.tensor(det[3:]), torch.tensor(gt[3:])).item()
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou > iou_thr:
                # find original index in ground_truths
                orig = ground_truths.index(gts[best_j])
                marker = gt_per_img[det[0]]
                local_j = sum(1 for g in ground_truths[:orig] if g[0] == det[0])
                if marker[local_j] == 0:
                    TP[i] = 1
                    marker[local_j] = 1
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        TP_c = torch.cumsum(TP, 0)
        FP_c = torch.cumsum(FP, 0)
        recalls = TP_c / (total_gt + eps)
        precisions = TP_c / (TP_c + FP_c + eps)
        precisions = torch.cat([torch.tensor([1.0]), precisions])
        recalls = torch.cat([torch.tensor([0.0]), recalls])
        APs.append(torch.trapz(precisions, recalls).item())
    return sum(APs) / len(APs) if APs else 0.0


def get_bboxes(loader, model, iou_thr=0.5, conf_thr=0.01, device="cuda", S=7, B=2, C=20):
    model.eval()
    all_pred, all_true = [], []
    idx = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_boxes = cells_to_boxes(preds, S, B, C)
            true_boxes = cells_to_boxes(targets, S, B, C)
            for i in range(imgs.size(0)):
                kept = nms(pred_boxes[i].tolist(), iou_thr, conf_thr)
                for b in kept:
                    all_pred.append([idx] + b)
                for b in true_boxes[i].tolist():
                    if b[1] > 0:  # has object
                        all_true.append([idx] + b)
                idx += 1
    model.train()
    return all_pred, all_true
