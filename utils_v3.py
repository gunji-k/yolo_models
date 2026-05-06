"""Decode YOLOv3 multi-scale predictions and extract bboxes for mAP.

Class scoring uses per-class sigmoid (v3, sec 2.2) — argmax across classes
gives the top label for VOC-style single-class assignment downstream.
"""
import torch

from utils import nms


def decode_v3(preds_list, anchors_norm):
    """preds_list: list of `num_scales` tensors [N, Si, Si, B, 5+C].
    anchors_norm: [num_scales, B, 2] in normalized [0, 1] coords.
    Returns [N, total_boxes, 6] = (cls, conf, cx, cy, w, h) in normalized coords.
    """
    if not isinstance(anchors_norm, torch.Tensor):
        anchors_norm = torch.tensor(anchors_norm, dtype=torch.float32)

    N = preds_list[0].shape[0]
    out_list = []
    for s_idx, preds in enumerate(preds_list):
        device = preds.device
        _, S, _, B, _ = preds.shape
        anchors = anchors_norm[s_idx].to(device) * S  # grid units

        cx = torch.arange(S, device=device).view(1, 1, S, 1).float()
        cy = torch.arange(S, device=device).view(1, S, 1, 1).float()
        pw = anchors[:, 0].view(1, 1, 1, B)
        ph = anchors[:, 1].view(1, 1, 1, B)

        bx = (torch.sigmoid(preds[..., 0]) + cx) / S
        by = (torch.sigmoid(preds[..., 1]) + cy) / S
        bw = pw * torch.exp(preds[..., 2].clamp(max=10)) / S
        bh = ph * torch.exp(preds[..., 3].clamp(max=10)) / S
        obj = torch.sigmoid(preds[..., 4])
        cls_probs = torch.sigmoid(preds[..., 5:])
        cls_score, cls_id = cls_probs.max(dim=-1)
        score = obj * cls_score

        out = torch.stack([cls_id.float(), score, bx, by, bw, bh], dim=-1)
        out_list.append(out.view(N, S * S * B, 6))
    return torch.cat(out_list, dim=1)


def get_bboxes_v3(loader, model, anchors_norm, iou_thr=0.5, conf_thr=0.01, device="cuda"):
    """Run model over loader, NMS, return (pred_boxes, true_boxes) in mAP format:
    [img_idx, cls, conf, cx, cy, w, h]."""
    model.eval()
    all_pred, all_true = [], []
    img_idx = 0
    with torch.no_grad():
        for imgs, target_boxes in loader:
            imgs = imgs.to(device)
            preds_list = model(imgs)
            decoded = decode_v3(preds_list, anchors_norm).cpu()  # [N, K, 6]
            for n in range(imgs.size(0)):
                kept = nms(decoded[n].tolist(), iou_thr=iou_thr, conf_thr=conf_thr)
                for b in kept:
                    all_pred.append([img_idx] + b)
                for row in target_boxes[n].tolist():
                    cls = row[0]
                    if cls < 0:
                        continue
                    all_true.append([img_idx, cls, 1.0, row[1], row[2], row[3], row[4]])
                img_idx += 1
    model.train()
    return all_pred, all_true
