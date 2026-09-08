import torch
from torch.nn import functional as F

from detectron2.layers import cat
from detectron2.modeling.poolers import ROIPooler


def build_blender(cfg):
    return Blender(cfg)


def boundary_weighted_mask_loss(pred_logits, gt_masks, boundary_weight,
                                 boundary_kernel=3):
    """BR1-lite（M6.5）：边界加权的实例 mask BCE。

    在 GT mask 的形态学边界带（dilate − erode，maxpool 实现）内把 BCE 权重
    提到 (1 + boundary_weight)，带外保持 1，再按逐实例加权均值聚合——与原
    mean(dim=-1) 的量级可比，仅重新分配像素间相对权重（零参数）。

    Args:
        pred_logits: (N, H*W) 实例 mask logits
        gt_masks: (N, H*W) 0/1 GT
        boundary_weight: 边界带内额外权重 λ；0 时严格退化为原 BCE 均值
        boundary_kernel: 形态学核大小（奇数）
    Returns:
        (N,) 每实例加权平均损失
    """
    losses = F.binary_cross_entropy_with_logits(
        pred_logits, gt_masks.to(dtype=torch.float32), reduction="none")
    if boundary_weight <= 0:
        return losses.mean(dim=-1)
    N = gt_masks.size(0)
    res = int(gt_masks.size(1) ** 0.5)
    gm = gt_masks.view(N, 1, res, res).float()
    pad = boundary_kernel // 2
    dil = F.max_pool2d(gm, boundary_kernel, 1, pad)
    ero = -F.max_pool2d(-gm, boundary_kernel, 1, pad)
    band = (dil - ero).clamp(0, 1).view(N, -1)
    w = 1.0 + boundary_weight * band
    return (losses * w).sum(dim=-1) / w.sum(dim=-1).clamp(min=1.0)


class Blender(object):
    def __init__(self, cfg):

        # fmt: off
        self.pooler_resolution = cfg.MODEL.BLENDMASK.BOTTOM_RESOLUTION
        sampling_ratio         = cfg.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO
        pooler_type            = cfg.MODEL.BLENDMASK.POOLER_TYPE
        pooler_scales          = cfg.MODEL.BLENDMASK.POOLER_SCALES
        self.attn_size         = cfg.MODEL.BLENDMASK.ATTN_SIZE
        self.top_interp        = cfg.MODEL.BLENDMASK.TOP_INTERP
        num_bases              = cfg.MODEL.BASIS_MODULE.NUM_BASES
        self.boundary_weight   = cfg.MODEL.BLENDMASK.BOUNDARY_LOSS_WEIGHT
        self.boundary_kernel   = cfg.MODEL.BLENDMASK.BOUNDARY_KERNEL
        # fmt: on

        self.attn_len = num_bases * self.attn_size * self.attn_size

        self.pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=2)

    def __call__(self, bases, proposals, gt_instances):
        if gt_instances is not None:
            # training
            # reshape attns
            dense_info = proposals["instances"]
            attns = dense_info.top_feats
            pos_inds = dense_info.pos_inds
            if pos_inds.numel() == 0:
                return None, {"loss_mask": sum([x.sum() * 0 for x in attns]) + bases[0].sum() * 0}

            gt_inds = dense_info.gt_inds

            rois = self.pooler(bases, [x.gt_boxes for x in gt_instances])
            rois = rois[gt_inds]
            pred_mask_logits = self.merge_bases(rois, attns)

            # gen targets
            gt_masks = []
            for instances_per_image in gt_instances:
                if len(instances_per_image.gt_boxes.tensor) == 0:
                    continue
                gt_mask_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.gt_boxes.tensor, self.pooler_resolution
                ).to(device=pred_mask_logits.device)
                gt_masks.append(gt_mask_per_image)
            gt_masks = cat(gt_masks, dim=0)
            gt_masks = gt_masks[gt_inds]
            N = gt_masks.size(0)
            gt_masks = gt_masks.view(N, -1)

            gt_ctr = dense_info.gt_ctrs
            loss_denorm = proposals["loss_denorm"]
            # BR1-lite：边界加权 BCE（BOUNDARY_LOSS_WEIGHT=0 时严格退化为原版）
            per_inst = boundary_weighted_mask_loss(
                pred_mask_logits, gt_masks, self.boundary_weight,
                self.boundary_kernel)
            mask_loss = (per_inst * gt_ctr).sum() / loss_denorm
            return None, {"loss_mask": mask_loss}
        else:
            # no proposals
            total_instances = sum([len(x) for x in proposals])
            if total_instances == 0:
                # add empty pred_masks results
                for box in proposals:
                    box.pred_masks = box.pred_classes.view(
                        -1, 1, self.pooler_resolution, self.pooler_resolution)
                return proposals, {}
            rois = self.pooler(bases, [x.pred_boxes for x in proposals])
            attns = cat([x.top_feat for x in proposals], dim=0)
            pred_mask_logits = self.merge_bases(rois, attns).sigmoid()
            pred_mask_logits = pred_mask_logits.view(
                -1, 1, self.pooler_resolution, self.pooler_resolution)
            start_ind = 0
            for box in proposals:
                end_ind = start_ind + len(box)
                box.pred_masks = pred_mask_logits[start_ind:end_ind]
                start_ind = end_ind
            return proposals, {}

    def merge_bases(self, rois, coeffs, location_to_inds=None):
        # merge predictions
        N = coeffs.size(0)
        if location_to_inds is not None:
            rois = rois[location_to_inds]
        N, B, H, W = rois.size()

        coeffs = coeffs.view(N, -1, self.attn_size, self.attn_size)
        coeffs = F.interpolate(coeffs, (H, W),
                               mode=self.top_interp).softmax(dim=1)
        masks_preds = (rois * coeffs).sum(dim=1)
        return masks_preds.view(N, -1)
