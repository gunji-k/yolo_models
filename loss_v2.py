"""YOLO v2 multi-part loss with anchor priors and dynamic IoU-based ignore mask.

Predictions are raw t_x, t_y, t_w, t_h, t_o, t_class for every (cell, anchor):
    bx = sigmoid(t_x) + cx       (grid units)
    by = sigmoid(t_y) + cy
    bw = pw * exp(t_w)           (grid units; pw, ph are anchor priors in cells)
    bh = ph * exp(t_h)
    obj = sigmoid(t_o)            (target = IoU of decoded box and matched GT)
    class = softmax(t_class)      (per-anchor)

Targets are computed on the fly from a padded ground-truth tensor of shape
[N, max_objs, 5] = (cls, cx, cy, w, h) in normalized [0, 1] coords. This lets
S vary per batch (multi-scale training).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_iou_xywh(b1, b2):
    """b1: [N, 4], b2: [M, 4] (cx, cy, w, h). Returns [N, M]."""
    b1_x1 = b1[:, 0:1] - b1[:, 2:3] / 2
    b1_y1 = b1[:, 1:2] - b1[:, 3:4] / 2
    b1_x2 = b1[:, 0:1] + b1[:, 2:3] / 2
    b1_y2 = b1[:, 1:2] + b1[:, 3:4] / 2
    b2_x1 = (b2[:, 0] - b2[:, 2] / 2).unsqueeze(0)
    b2_y1 = (b2[:, 1] - b2[:, 3] / 2).unsqueeze(0)
    b2_x2 = (b2[:, 0] + b2[:, 2] / 2).unsqueeze(0)
    b2_y2 = (b2[:, 1] + b2[:, 3] / 2).unsqueeze(0)
    inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    a1 = (b1[:, 2:3] * b1[:, 3:4])
    a2 = ((b2[:, 2] * b2[:, 3])).unsqueeze(0)
    return inter / (a1 + a2 - inter + 1e-9)


def _wh_iou(wh_boxes, wh_anchors):
    """Shape-only IoU between (w, h) boxes and (w, h) anchors, both centered at origin.
    wh_boxes: [G, 2], wh_anchors: [B, 2] -> [G, B]."""
    inter_w = torch.minimum(wh_boxes[:, None, 0], wh_anchors[None, :, 0])
    inter_h = torch.minimum(wh_boxes[:, None, 1], wh_anchors[None, :, 1])
    inter = inter_w * inter_h
    a_box = (wh_boxes[:, 0] * wh_boxes[:, 1])[:, None]
    a_anc = (wh_anchors[:, 0] * wh_anchors[:, 1])[None, :]
    return inter / (a_box + a_anc - inter + 1e-9)


class YoloV2Loss(nn.Module):
    def __init__(
        self,
        anchors,
        num_classes=20,
        lambda_coord=5.0,
        lambda_obj=1.0,
        lambda_noobj=1.0,
        lambda_class=1.0,
        ignore_iou_threshold=0.6,
    ):
        super().__init__()
        if not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anchors)  # [B, 2] in grid units
        self.B = anchors.shape[0]
        self.C = num_classes
        self.lc = lambda_coord
        self.lo = lambda_obj
        self.ln = lambda_noobj
        self.lcls = lambda_class
        self.ignore_iou_threshold = ignore_iou_threshold

    def forward(self, preds, target_boxes):
        """preds: [N, S, S, B, 5+C] raw outputs.
        target_boxes: [N, max_objs, 5] (cls, cx, cy, w, h) normalized; cls=-1 = padding."""
        N, S, _, B, _ = preds.shape
        assert B == self.B
        device = preds.device
        anchors = self.anchors.to(device)

        tx = preds[..., 0]
        ty = preds[..., 1]
        tw = preds[..., 2]
        th = preds[..., 3]
        to = preds[..., 4]
        tcls = preds[..., 5:]  # [N, S, S, B, C]

        # Decode to grid units (used for IoU against GT in obj target & ignore mask).
        cx_grid = torch.arange(S, device=device).view(1, 1, S, 1).expand(N, S, S, B).float()
        cy_grid = torch.arange(S, device=device).view(1, S, 1, 1).expand(N, S, S, B).float()
        pw = anchors[:, 0].view(1, 1, 1, B).expand(N, S, S, B)
        ph = anchors[:, 1].view(1, 1, 1, B).expand(N, S, S, B)

        bx = torch.sigmoid(tx) + cx_grid
        by = torch.sigmoid(ty) + cy_grid
        bw = pw * torch.exp(tw.clamp(max=10))  # cap exp() for numerical safety
        bh = ph * torch.exp(th.clamp(max=10))

        sxy = torch.stack([torch.sigmoid(tx), torch.sigmoid(ty)], dim=-1)  # [N,S,S,B,2]
        twh = torch.stack([tw, th], dim=-1)
        sto = torch.sigmoid(to)

        # Allocate target/mask buffers.
        obj_mask = torch.zeros(N, S, S, B, dtype=torch.bool, device=device)
        noobj_mask = torch.ones(N, S, S, B, dtype=torch.bool, device=device)
        txy_target = torch.zeros(N, S, S, B, 2, device=device)
        twh_target = torch.zeros(N, S, S, B, 2, device=device)
        cls_target = torch.zeros(N, S, S, B, dtype=torch.long, device=device)
        iou_target = torch.zeros(N, S, S, B, device=device)

        pred_boxes_grid = torch.stack([bx, by, bw, bh], dim=-1).view(N, -1, 4)

        for n in range(N):
            valid = target_boxes[n, :, 0] >= 0
            if not valid.any():
                continue
            gt = target_boxes[n][valid].to(device)  # [G, 5]
            gt_cls = gt[:, 0].long()
            gx = gt[:, 1] * S
            gy = gt[:, 2] * S
            gw = gt[:, 3] * S
            gh = gt[:, 4] * S
            G = gt.shape[0]

            # Ignore mask: predicted boxes whose max IoU with any GT exceeds threshold.
            gt_grid = torch.stack([gx, gy, gw, gh], dim=-1)  # [G, 4]
            with torch.no_grad():
                pred_n = pred_boxes_grid[n]  # [S*S*B, 4]
                ious_pred_gt = _pairwise_iou_xywh(pred_n, gt_grid)  # [S*S*B, G]
                max_ious, _ = ious_pred_gt.max(dim=1)
                ignore = (max_ious > self.ignore_iou_threshold).view(S, S, B)
            noobj_mask[n] = ~ignore  # default = not ignored; will be cleared at responsible cells too

            # Anchor assignment per GT: pick anchor with highest shape IoU at GT's cell.
            anchor_iou = _wh_iou(torch.stack([gw, gh], dim=-1), anchors)  # [G, B]
            best_anchor = anchor_iou.argmax(dim=1)  # [G]
            cells_j = gx.long().clamp(0, S - 1)
            cells_i = gy.long().clamp(0, S - 1)

            for g in range(G):
                ci = int(cells_i[g])
                cj = int(cells_j[g])
                ba = int(best_anchor[g])
                obj_mask[n, ci, cj, ba] = True
                noobj_mask[n, ci, cj, ba] = False

                txy_target[n, ci, cj, ba, 0] = gx[g] - cj
                txy_target[n, ci, cj, ba, 1] = gy[g] - ci
                twh_target[n, ci, cj, ba, 0] = torch.log(gw[g] / anchors[ba, 0] + 1e-9)
                twh_target[n, ci, cj, ba, 1] = torch.log(gh[g] / anchors[ba, 1] + 1e-9)
                cls_target[n, ci, cj, ba] = gt_cls[g]

                with torch.no_grad():
                    pb = torch.tensor(
                        [bx[n, ci, cj, ba], by[n, ci, cj, ba],
                         bw[n, ci, cj, ba], bh[n, ci, cj, ba]],
                        device=device,
                    ).unsqueeze(0)
                    gb = gt_grid[g].unsqueeze(0)
                    iou_target[n, ci, cj, ba] = _pairwise_iou_xywh(pb, gb).squeeze()

        # Coordinate loss (only at responsible cells/anchors).
        if obj_mask.any():
            xy_loss = F.mse_loss(sxy[obj_mask], txy_target[obj_mask], reduction="sum")
            wh_loss = F.mse_loss(twh[obj_mask], twh_target[obj_mask], reduction="sum")
            obj_loss = F.mse_loss(sto[obj_mask], iou_target[obj_mask], reduction="sum")
            cls_loss = F.cross_entropy(tcls[obj_mask], cls_target[obj_mask], reduction="sum")
        else:
            xy_loss = wh_loss = obj_loss = cls_loss = preds.sum() * 0.0

        noobj_loss = F.mse_loss(
            sto[noobj_mask], torch.zeros_like(sto[noobj_mask]), reduction="sum"
        )

        total = (
            self.lc * (xy_loss + wh_loss)
            + self.lo * obj_loss
            + self.ln * noobj_loss
            + self.lcls * cls_loss
        ) / N
        return total
