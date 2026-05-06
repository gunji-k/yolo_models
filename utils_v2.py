"""Decoder + bbox extraction for YOLO v2 predictions.

Reuses the v1 NMS and mAP from utils.py.
"""
import torch

from utils import nms


def decode_v2(preds, anchors):
    """preds: [N, S, S, B, 5+C] raw outputs. anchors: [B, 2] in grid units.
    Returns [N, S*S*B, 6] = (cls, conf, cx, cy, w, h) in normalized coords."""
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, dtype=torch.float32)
    anchors = anchors.to(preds.device)

    N, S, _, B, _ = preds.shape
    device = preds.device

    cx = torch.arange(S, device=device).view(1, 1, S, 1).float()
    cy = torch.arange(S, device=device).view(1, S, 1, 1).float()
    pw = anchors[:, 0].view(1, 1, 1, B)
    ph = anchors[:, 1].view(1, 1, 1, B)

    bx = (torch.sigmoid(preds[..., 0]) + cx) / S
    by = (torch.sigmoid(preds[..., 1]) + cy) / S
    bw = pw * torch.exp(preds[..., 2].clamp(max=10)) / S
    bh = ph * torch.exp(preds[..., 3].clamp(max=10)) / S
    obj = torch.sigmoid(preds[..., 4])
    cls_probs = torch.softmax(preds[..., 5:], dim=-1)
    cls_score, cls_id = cls_probs.max(dim=-1)
    score = obj * cls_score

    out = torch.stack([cls_id.float(), score, bx, by, bw, bh], dim=-1)
    return out.view(N, S * S * B, 6)


def get_bboxes_v2(loader, model, anchors, iou_thr=0.5, conf_thr=0.01, device="cuda"):
    """Run model over loader, NMS predictions, return (pred_boxes, true_boxes)
    in the format expected by utils.mean_average_precision:
        [img_idx, cls, conf, cx, cy, w, h]
    """
    model.eval()
    all_pred, all_true = [], []
    img_idx = 0
    with torch.no_grad():
        for imgs, target_boxes in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            decoded = decode_v2(preds, anchors).cpu()  # [N, K, 6]
            for n in range(imgs.size(0)):
                kept = nms(decoded[n].tolist(), iou_thr=iou_thr, conf_thr=conf_thr)
                for b in kept:
                    all_pred.append([img_idx] + b)
                # Ground truth from padded boxes tensor.
                for row in target_boxes[n].tolist():
                    cls = row[0]
                    if cls < 0:  # padding
                        continue
                    # GT format: [img_idx, cls, conf=1.0, cx, cy, w, h]
                    all_true.append([img_idx, cls, 1.0, row[1], row[2], row[3], row[4]])
                img_idx += 1
    model.train()
    return all_pred, all_true
