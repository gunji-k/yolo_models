"""YOLOv3 multi-scale loss (arXiv 1804.02767, sec 2.1-2.2).

Differences vs YoloV2Loss:
- Predictions arrive as a list of 3 tensors, one per scale.
- Each GT is assigned to one (scale, anchor) by best shape-IoU across all 9 anchors.
- Objectness target is a hard 1 for the responsible (scale, cell, anchor),
  trained with BCE-with-logits. v2 used IoU-as-target with MSE.
- Class loss is per-class sigmoid + BCE against one-hot (multi-label, sec 2.2).
- Per-scale ignore band: predictions with decoded IoU > 0.5 against any GT
  at that scale are excluded from the no-object loss.
- Coordinate loss: xy in sigmoid space (MSE between sigmoid(tx) and the in-cell
  offset in [0, 1]) and wh in raw t-space (MSE between tw and log(gw/pw)). This
  matches the Darknet reference; computing xy MSE in raw t-space against
  logit(offset) blows up at cell edges.

Anchors are passed as [3 scales, B, 2] in normalized [0, 1] coords; converted
to that scale's grid units (multiply by S) at runtime.
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
    a1 = b1[:, 2:3] * b1[:, 3:4]
    a2 = (b2[:, 2] * b2[:, 3]).unsqueeze(0)
    return inter / (a1 + a2 - inter + 1e-9)


def _wh_iou(wh_boxes, wh_anchors):
    """Shape-only IoU. wh_boxes: [G, 2], wh_anchors: [K, 2] -> [G, K]."""
    inter_w = torch.minimum(wh_boxes[:, None, 0], wh_anchors[None, :, 0])
    inter_h = torch.minimum(wh_boxes[:, None, 1], wh_anchors[None, :, 1])
    inter = inter_w * inter_h
    a_box = (wh_boxes[:, 0] * wh_boxes[:, 1])[:, None]
    a_anc = (wh_anchors[:, 0] * wh_anchors[:, 1])[None, :]
    return inter / (a_box + a_anc - inter + 1e-9)


class YoloV3Loss(nn.Module):
    def __init__(
        self,
        anchors,
        num_classes=20,
        lambda_coord=5.0,
        lambda_obj=1.0,
        lambda_noobj=1.0,
        lambda_class=1.0,
        ignore_iou_threshold=0.5,
    ):
        super().__init__()
        if not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(anchors, dtype=torch.float32)
        assert anchors.dim() == 3 and anchors.shape[-1] == 2
        self.register_buffer("anchors_norm", anchors)  # [num_scales, B, 2]
        self.num_scales = anchors.shape[0]
        self.B = anchors.shape[1]
        self.C = num_classes
        self.lc = lambda_coord
        self.lo = lambda_obj
        self.ln = lambda_noobj
        self.lcls = lambda_class
        self.ignore_iou_threshold = ignore_iou_threshold

    def forward(self, preds_list, target_boxes):
        """preds_list: list of `num_scales` tensors [N, Si, Si, B, 5+C].
        target_boxes: [N, max_objs, 5] (cls, cx, cy, w, h) normalized; cls=-1 = padding.
        """
        assert len(preds_list) == self.num_scales
        device = preds_list[0].device
        N = preds_list[0].shape[0]

        # Flat [num_scales*B, 2] for cross-scale anchor matching.
        all_anchors_norm = self.anchors_norm.to(device).view(-1, 2)

        # Per-scale: decode preds + allocate target buffers.
        scale_data = []
        for s_idx, preds in enumerate(preds_list):
            _, S, _, B, _ = preds.shape
            assert B == self.B
            anchors_grid = self.anchors_norm[s_idx].to(device) * S
            cx = torch.arange(S, device=device).view(1, 1, S, 1).expand(N, S, S, B).float()
            cy = torch.arange(S, device=device).view(1, S, 1, 1).expand(N, S, S, B).float()
            pw = anchors_grid[:, 0].view(1, 1, 1, B).expand(N, S, S, B)
            ph = anchors_grid[:, 1].view(1, 1, 1, B).expand(N, S, S, B)

            tx, ty = preds[..., 0], preds[..., 1]
            tw, th = preds[..., 2], preds[..., 3]
            to = preds[..., 4]
            tcls = preds[..., 5:]

            bx = torch.sigmoid(tx) + cx
            by = torch.sigmoid(ty) + cy
            bw = pw * torch.exp(tw.clamp(max=10))
            bh = ph * torch.exp(th.clamp(max=10))

            scale_data.append({
                "S": S, "preds": preds,
                "tx": tx, "ty": ty, "tw": tw, "th": th, "to": to, "tcls": tcls,
                "bx": bx, "by": by, "bw": bw, "bh": bh,
                "anchors_grid": anchors_grid,
                "obj_mask": torch.zeros(N, S, S, B, dtype=torch.bool, device=device),
                "noobj_mask": torch.ones(N, S, S, B, dtype=torch.bool, device=device),
                "txy_target": torch.zeros(N, S, S, B, 2, device=device),
                "twh_target": torch.zeros(N, S, S, B, 2, device=device),
                "cls_target": torch.zeros(N, S, S, B, dtype=torch.long, device=device),
            })

        for n in range(N):
            valid = target_boxes[n, :, 0] >= 0
            if not valid.any():
                continue
            gt = target_boxes[n][valid].to(device)  # [G, 5] normalized
            G = gt.shape[0]
            gt_cls = gt[:, 0].long()
            gt_xywh_norm = gt[:, 1:]
            gt_wh_norm = gt[:, 3:5]

            # Per-scale ignore band on decoded predictions.
            for s_idx, sd in enumerate(scale_data):
                S = sd["S"]
                gt_grid = gt_xywh_norm * S
                with torch.no_grad():
                    pred_n = torch.stack(
                        [sd["bx"][n], sd["by"][n], sd["bw"][n], sd["bh"][n]], dim=-1
                    ).view(-1, 4)
                    ious = _pairwise_iou_xywh(pred_n, gt_grid)  # [S*S*B, G]
                    max_iou, _ = ious.max(dim=1)
                    ignore = (max_iou > self.ignore_iou_threshold).view(S, S, self.B)
                sd["noobj_mask"][n] = ~ignore

            # Cross-scale assignment: best shape-IoU anchor (over all 9) per GT.
            anchor_iou = _wh_iou(gt_wh_norm, all_anchors_norm)  # [G, num_scales*B]
            best_flat = anchor_iou.argmax(dim=1)
            best_scale = (best_flat // self.B).tolist()
            best_in_scale = (best_flat % self.B).tolist()

            for g in range(G):
                s_idx = best_scale[g]
                ba = best_in_scale[g]
                sd = scale_data[s_idx]
                S = sd["S"]
                gx = gt_xywh_norm[g, 0] * S
                gy = gt_xywh_norm[g, 1] * S
                gw = gt_xywh_norm[g, 2] * S
                gh = gt_xywh_norm[g, 3] * S
                cj = int(gx.long().clamp(0, S - 1))
                ci = int(gy.long().clamp(0, S - 1))

                sd["obj_mask"][n, ci, cj, ba] = True
                sd["noobj_mask"][n, ci, cj, ba] = False
                sd["txy_target"][n, ci, cj, ba, 0] = gx - cj
                sd["txy_target"][n, ci, cj, ba, 1] = gy - ci
                anc = sd["anchors_grid"][ba]
                sd["twh_target"][n, ci, cj, ba, 0] = torch.log(gw / anc[0] + 1e-9)
                sd["twh_target"][n, ci, cj, ba, 1] = torch.log(gh / anc[1] + 1e-9)
                sd["cls_target"][n, ci, cj, ba] = gt_cls[g]

        zero = preds_list[0].sum() * 0.0  # zero with grad
        xy_total = zero.clone()
        wh_total = zero.clone()
        obj_total = zero.clone()
        noobj_total = zero.clone()
        cls_total = zero.clone()

        for sd in scale_data:
            obj_mask = sd["obj_mask"]
            noobj_mask = sd["noobj_mask"]

            sxy = torch.stack([torch.sigmoid(sd["tx"]), torch.sigmoid(sd["ty"])], dim=-1)
            twh = torch.stack([sd["tw"], sd["th"]], dim=-1)

            if obj_mask.any():
                # xy: MSE between sigmoid(tx) and in-cell offset in [0, 1].
                # wh: MSE between tw and log(gw/pw).
                xy_loss = F.mse_loss(sxy[obj_mask], sd["txy_target"][obj_mask], reduction="sum")
                wh_loss = F.mse_loss(twh[obj_mask], sd["twh_target"][obj_mask], reduction="sum")
                obj_loss = F.binary_cross_entropy_with_logits(
                    sd["to"][obj_mask], torch.ones_like(sd["to"][obj_mask]), reduction="sum",
                )
                cls_one_hot = F.one_hot(sd["cls_target"][obj_mask], num_classes=self.C).float()
                cls_loss = F.binary_cross_entropy_with_logits(
                    sd["tcls"][obj_mask], cls_one_hot, reduction="sum",
                )
            else:
                xy_loss = wh_loss = obj_loss = cls_loss = sd["preds"].sum() * 0.0

            if noobj_mask.any():
                noobj_loss = F.binary_cross_entropy_with_logits(
                    sd["to"][noobj_mask],
                    torch.zeros_like(sd["to"][noobj_mask]),
                    reduction="sum",
                )
            else:
                noobj_loss = sd["preds"].sum() * 0.0

            xy_total = xy_total + xy_loss
            wh_total = wh_total + wh_loss
            obj_total = obj_total + obj_loss
            noobj_total = noobj_total + noobj_loss
            cls_total = cls_total + cls_loss

        total = (
            self.lc * (xy_total + wh_total)
            + self.lo * obj_total
            + self.ln * noobj_total
            + self.lcls * cls_total
        ) / N

        components = {
            "xy": (self.lc * xy_total / N).detach(),
            "wh": (self.lc * wh_total / N).detach(),
            "obj": (self.lo * obj_total / N).detach(),
            "noobj": (self.ln * noobj_total / N).detach(),
            "cls": (self.lcls * cls_total / N).detach(),
        }
        return total, components
