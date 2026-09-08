# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head

from .blender import build_blender
from .basis_module import build_basis_module

__all__ = ["BlendMask"]


@META_ARCH_REGISTRY.register()
class BlendMask(nn.Module):
    """
    Main class for BlendMask architectures (see https://arxiv.org/abd/1901.02446).
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.instance_loss_weight = cfg.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.blender = build_blender(cfg)
        self.basis_module = build_basis_module(cfg, self.backbone.output_shape())

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
        if self.combine_on:
            self.panoptic_module = build_sem_seg_head(cfg, self.backbone.output_shape())
            self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
            self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
            self.combine_instances_confidence_threshold = (
                cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH)

        # build top module
        # 原实现直接读 cfg.MODEL.FPN.OUT_CHANNELS，但用 BiFPN 时实际通道数由
        # MODEL.BiFPN.OUT_CHANNELS 决定（默认 160，与 FPN 默认的 256 不同），
        # 会导致 top_layer 与 backbone 输出通道不匹配。这里直接向 backbone 查询
        # 真实通道数，使 FPN / BiFPN 都能自动适配。
        in_channels = self.backbone.output_shape()[
            cfg.MODEL.FCOS.IN_FEATURES[0]].channels
        num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES
        attn_size = cfg.MODEL.BLENDMASK.ATTN_SIZE
        attn_len = num_bases * attn_size * attn_size
        self.top_layer = nn.Conv2d(
            in_channels, attn_len,
            kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.top_layer.weight, std=0.01)
        torch.nn.init.constant_(self.top_layer.bias, 0)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # KD1（M6.5）：检测+mask 蒸馏。WEIGHTS 为空时完全关闭（零行为差异）。
        self.distill_cfg = {
            "weights": cfg.MODEL.DISTILL.WEIGHTS,
            "teacher_opts": cfg.MODEL.DISTILL.TEACHER_OPTS,
            "w_cls": cfg.MODEL.DISTILL.W_CLS,
            "w_reg": cfg.MODEL.DISTILL.W_REG,
            "w_bases": cfg.MODEL.DISTILL.W_BASES,
            "fg_thresh": cfg.MODEL.DISTILL.FG_THRESH,
        }
        self.teacher = None
        if cfg.MODEL.DISTILL.WEIGHTS:
            self._build_teacher(cfg)
        self.to(self.device)

    def _build_teacher(self, cfg):
        """构建冻结的 teacher（同 META_ARCHITECTURE，覆盖 TEACHER_OPTS 指定的
        架构差异如 VIG.VERSION=b），加载 MODEL.DISTILL.WEIGHTS。"""
        from detectron2.checkpoint import DetectionCheckpointer
        tcfg = cfg.clone()
        tcfg.defrost()
        opts = self.distill_cfg["teacher_opts"]
        if opts:
            # 兼容两种格式：YAML 风格 KEY=VALUE（shell 安全，推荐）与
            # merge_from_list 风格 KEY VALUE（空格分隔，仅在代码内构造时用）
            tokens = opts.split()
            if "=" in tokens[0]:
                pairs = []
                for tok in tokens:
                    key, _, val = tok.partition("=")
                    pairs.extend([key, val])
            else:
                pairs = tokens
            tcfg.merge_from_list(pairs)
        # teacher 与 student 输入分辨率一致（由 run 脚本保证），此处只覆盖
        # checkpoint 路径本身，避免 student 的 WEIGHTS（通常为空）串扰。
        tcfg.MODEL.WEIGHTS = ""
        # 构建时递归保护：teacher 构建中不再创建 teacher
        saved = cfg.MODEL.DISTILL.WEIGHTS
        try:
            tcfg.MODEL.DISTILL.WEIGHTS = ""
            teacher = META_ARCH_REGISTRY.get(tcfg.MODEL.META_ARCHITECTURE)(tcfg)
        finally:
            tcfg.MODEL.DISTILL.WEIGHTS = saved
        DetectionCheckpointer(teacher).load(self.distill_cfg["weights"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        logger = logging.getLogger(__name__)
        logger.info("KD1 teacher built from {} (opts: {})".format(
            self.distill_cfg["weights"], self.distill_cfg["teacher_opts"]))

    def _distill_losses(self, batched_inputs, images, features):
        """teacher 前向 + 三路蒸馏损失（cls sigmoid / reg L1 / bases 特征 MSELoss）。

        teacher 与 student 共享同一 ImageList（同 padding、同输入分辨率），因此
        FPN 各层像素一一对应；bases 层按 pooler 空间对齐（teacher/student 的
        basis tower 输出分辨率一致，均为 in_features[0] 上采样 2×）。
        """
        cfg_w = self.distill_cfg
        with torch.no_grad():
            # m1.train() 会级联把 teacher 也置为 train 模式（BN 统计漂移），
            # 每次蒸馏前强制 eval，保证 teacher 行为恒定。
            self.teacher.eval()
            t_images = [x["image"].to(self.device) for x in batched_inputs]
            t_images = [self.teacher.normalizer(x) for x in t_images]
            t_images = ImageList.from_tensors(
                t_images, self.teacher.backbone.size_divisibility)
            # 与 student 对齐到同一 padded 尺寸（teacher/student 分辨率一致时天然相同）
            if (t_images.tensor.shape[-2:] != images.tensor.shape[-2:]):
                return {}
            t_features = self.teacher.backbone(t_images.tensor)
            t_logits, t_reg, t_ctr, t_top, _ = self.teacher.proposal_generator.forward_head(
                t_features, self.teacher.top_layer)
            t_basis_out, _ = self.teacher.basis_module(t_features)

        # student 侧：重用主 forward 已算的 features，但 head 输出需重新取
        # （主 forward 走 proposal_generator 内部，这里直接调 forward_head）
        s_logits, s_reg, s_ctr, s_top, _ = self.proposal_generator.forward_head(
            features, self.top_layer)
        s_basis_out, _ = self.basis_module(features)

        losses = {}
        # 1) 分类蒸馏：sigmoid 后 BCE（对概率蒸馏，量级稳定）
        cls_loss = 0.0
        for tl, sl in zip(t_logits, s_logits):
            cls_loss = cls_loss + F.mse_loss(sl.sigmoid(), tl.sigmoid())
        losses["loss_kd_cls"] = cls_loss * cfg_w["w_cls"]
        # 2) 回归蒸馏：L1（relu 后的距离图，逐层）
        reg_loss = 0.0
        for tr, sr in zip(t_reg, s_reg):
            reg_loss = reg_loss + F.l1_loss(sr, tr)
        losses["loss_kd_reg"] = reg_loss * cfg_w["w_reg"]
        # 3) bases 特征蒸馏：teacher 前景区域加权 MSELoss
        bases_loss = 0.0
        for t_b, s_b in zip(t_basis_out["bases"], s_basis_out["bases"]):
            if t_b.shape != s_b.shape:
                t_b = F.interpolate(t_b, s_b.shape[-2:], mode="bilinear",
                                    align_corners=False)
            # 前景质量图：teacher bases 的均值幅度作为逐像素权重
            if cfg_w["fg_thresh"] > 0:
                w = (t_b.abs().mean(dim=1, keepdim=True) > cfg_w["fg_thresh"]).float()
                denom = w.sum().clamp(min=1.0)
                bases_loss = bases_loss + (F.mse_loss(s_b, t_b, reduction="none") * w).sum() / denom
            else:
                bases_loss = bases_loss + F.mse_loss(s_b, t_b)
        losses["loss_kd_bases"] = bases_loss * cfg_w["w_bases"]
        return losses

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            instances: Instances
            sem_seg: semantic segmentation ground truth.
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: each dict is the results for one image. The dict
                contains the following keys:
                "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                    See the return value of
                    :func:`combine_semantic_and_instance_outputs` for its format.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if self.combine_on:
            if "sem_seg" in batched_inputs[0]:
                gt_sem = [x["sem_seg"].to(self.device) for x in batched_inputs]
                gt_sem = ImageList.from_tensors(
                    gt_sem, self.backbone.size_divisibility, self.panoptic_module.ignore_value
                ).tensor
            else:
                gt_sem = None
            sem_seg_results, sem_seg_losses = self.panoptic_module(features, gt_sem)

        if "basis_sem" in batched_inputs[0]:
            basis_sem = [x["basis_sem"].to(self.device) for x in batched_inputs]
            basis_sem = ImageList.from_tensors(
                basis_sem, self.backbone.size_divisibility, 0).tensor
        else:
            basis_sem = None
        basis_out, basis_losses = self.basis_module(features, basis_sem)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances, self.top_layer)
        detector_results, detector_losses = self.blender(
            basis_out["bases"], proposals, gt_instances)

        if self.training:
            losses = {}
            losses.update(basis_losses)
            losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
            losses.update(proposal_losses)
            if self.teacher is not None:
                losses.update(self._distill_losses(batched_inputs, images, features))
            if self.combine_on:
                losses.update(sem_seg_losses)
            return losses

        processed_results = []
        for i, (detector_result, input_per_image, image_size) in enumerate(zip(
                detector_results, batched_inputs, images.image_sizes)):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            detector_r = detector_postprocess(detector_result, height, width)
            processed_result = {"instances": detector_r}
            if self.combine_on:
                sem_seg_r = sem_seg_postprocess(
                    sem_seg_results[i], image_size, height, width)
                processed_result["sem_seg"] = sem_seg_r
            if "seg_thing_out" in basis_out:
                seg_thing_r = sem_seg_postprocess(
                    basis_out["seg_thing_out"], image_size, height, width)
                processed_result["sem_thing_seg"] = seg_thing_r
            if self.basis_module.visualize:
                processed_result["bases"] = basis_out["bases"]
            processed_results.append(processed_result)

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold)
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results
