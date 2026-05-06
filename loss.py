"""YOLO v1 multi-part loss (Eq. 3 of the paper)."""
import torch
import torch.nn as nn
from utils import iou_xywh


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lc, self.ln = lambda_coord, lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):
        # pred/target: [N, S, S, C + 5B]. Target only populates box 0 slot.
        N, S, _, _ = pred.shape
        C, B = self.C, self.B

        # Split out predicted boxes
        pred_boxes = [pred[..., C + 1 + b * 5 : C + 5 + b * 5] for b in range(B)]
        pred_conf = [pred[..., C + b * 5 : C + 1 + b * 5] for b in range(B)]
        tgt_box = target[..., C + 1 : C + 5]
        obj_mask = target[..., C : C + 1]  # 1 if object in cell

        # IoU for each predicted box vs truth, pick responsible predictor
        ious = torch.stack([iou_xywh(pb, tgt_box) for pb in pred_boxes], dim=0)  # [B,N,S,S,1]
        best_iou, best_idx = ious.max(dim=0)  # [N,S,S,1]

        # Gather the "responsible" predicted box and its confidence
        stacked_boxes = torch.stack(pred_boxes, dim=0)  # [B,N,S,S,4]
        stacked_conf = torch.stack(pred_conf, dim=0)  # [B,N,S,S,1]
        idx_b = best_idx.unsqueeze(0).expand(1, *best_idx.shape[:-1], 4)
        resp_box = torch.gather(stacked_boxes, 0, idx_b).squeeze(0)
        resp_conf = torch.gather(stacked_conf, 0, best_idx.unsqueeze(0)).squeeze(0)

        # --- Coordinate loss (xy + sqrt(wh)) ---
        resp_xy = obj_mask * resp_box[..., 0:2]
        tgt_xy = obj_mask * tgt_box[..., 0:2]
        # sqrt-of-wh term: clamp pred wh to >= 0 so gradients stay finite at init
        resp_wh = resp_box[..., 2:4]
        resp_wh_s = obj_mask * torch.sqrt(resp_wh.clamp(min=0) + 1e-6)
        tgt_wh_s = obj_mask * torch.sqrt(tgt_box[..., 2:4] + 1e-6)
        coord_loss = self.mse(resp_xy, tgt_xy) + self.mse(resp_wh_s, tgt_wh_s)

        # --- Object confidence loss (target = IoU of responsible predictor) ---
        obj_loss = self.mse(obj_mask * resp_conf, obj_mask * best_iou.detach())

        # --- No-object loss (over all B predictors in empty cells) ---
        noobj_loss = 0.0
        for c in pred_conf:
            noobj_loss = noobj_loss + self.mse((1 - obj_mask) * c, torch.zeros_like(c))

        # --- Class loss ---
        class_loss = self.mse(obj_mask * pred[..., :C], obj_mask * target[..., :C])

        total = (
            self.lc * coord_loss
            + obj_loss
            + self.ln * noobj_loss
            + class_loss
        ) / N
        return total
