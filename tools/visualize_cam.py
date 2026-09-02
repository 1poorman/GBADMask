#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GBADMask / detectron2 模型热力图可视化与定量对比。

本脚本做两件事：

**1. 可视化** —— 画 CAM 热力图看"模型在看哪里"（Grad-CAM++ / Eigen-CAM / bases）。

**2. 定量对比** —— 用标准化指标给多个模型的热力图打分，用于论证某个模块
（例如本项目的 ProtoNetV2 / LSK）是否真的比基线更准。

设计上通过 ``ModelAdapter`` 抽象层适配不同 meta architecture，因此不只支持
本项目的 BlendMask，也能跑 detectron2 原生模型（Mask R-CNN / Faster R-CNN /
RetinaNet 等），只要模型带 FPN 式多尺度特征即可。

支持的模型
----------
* ``BlendMask`` / ``BlendMask2``（本项目 adet） —— 额外支持 bases 可视化
* ``GeneralizedRCNN``（Mask R-CNN / Faster R-CNN / Cascade R-CNN / PointRend）
* ``RetinaNet``
* 其他含 ``backbone`` 的模型 —— 走通用 fallback（仅 Eigen-CAM 与特征可视化）

定量指标（需要 GT 标注，加 ``--metric`` 开启）
--------------------------------------------
* ``EBPG`` —— Energy-Based Pointing Game：GT 区域内的 CAM 能量 / 全图总能量。
  **不需要阈值**，越高说明模型的注意力越集中在真正有物体的地方，是 CAM 评估里
  最常用的主指标。
* ``IoU@20%`` —— 取 CAM 值最高的 20% 像素二值化，与 GT mask 求 IoU。
* ``Hit@15%`` —— Pointing Game：CAM 值最高的 15% 像素中落在 GT 内的比例。
* ``BgRatio`` —— 背景区域平均激活 / 前景区域平均激活，越低越好（抑制背景的能力）。

用法
----
单模型可视化::

    conda activate gbadmask
    cd /home/huachenghao/codes/GBADMask
    python tools/visualize_cam.py \\
        --config-file configs/run-coco128-test.yaml \\
        --input datasets/coco128-seg/val2017/000000000529.jpg \\
        --output-dir output/cam \\
        MODEL.WEIGHTS output/coco128-test/model_final.pth

多模型定量对比（核心用途）::

    python tools/visualize_cam.py \\
        --models \\
          "baseline:configs/run-vig.yaml:output/base/model_final.pth" \\
          "ours:configs/run-BlendMask2-vig.yaml:output/ours/model_final.pth" \\
        --input "datasets/coco128-seg/val2017/*.jpg" \\
        --output-dir output/cmp --metric --all-layers

``--models`` 每项格式为 ``名称:配置文件:权重``，权重可留空表示随机初始化。
输出 ``metrics.csv``（每张图每个模型的指标）与 ``metrics_summary.txt``（汇总）。

detectron2 原生模型（需自备权重，配置文件可用其源码 configs 目录）::

    python tools/visualize_cam.py \\
        --models "maskrcnn:../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml:/path/to/model_final.pth" \\
        --input img.jpg --output-dir output/cmp
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 无显示环境下必须显式指定 Agg，否则 matplotlib 会尝试连 X server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.structures import ImageList, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode

from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer

logger = setup_logger(name="visualize_cam")


# =========================================================================== #
# 定量指标
# =========================================================================== #
def _gt_mask_from_anns(anns, height, width):
    """把 COCO 标注（polygon 或 RLE）合并成一张 (H, W) 的 0/1 前景图。"""
    mask = np.zeros((height, width), dtype=np.float32)
    for ann in anns:
        seg = ann.get("segmentation")
        if not seg:
            continue
        try:
            from pycocotools import mask as mask_utils
            if isinstance(seg, dict):               # RLE
                m = mask_utils.decode(seg)
            else:                                    # polygon（可能多段）
                rles = mask_utils.frPyObjects(seg, height, width)
                m = mask_utils.decode(rles)
                if m.ndim == 3:
                    m = m.any(axis=2)
            mask = np.maximum(mask, m.astype(np.float32))
        except Exception:
            # 退化到 bbox：没有 pycocotools 或标注异常时仍能给出粗略前景
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            x0, y0 = int(max(0, x)), int(max(0, y))
            x1, y1 = int(min(width, x + w)), int(min(height, y + h))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1.0
    return mask


def compute_metrics(cam, gt_mask):
    """在 GT mask 上评估 CAM 质量。

    Args:
        cam: (Hc, Wc) float，已归一化到 [0, 1]
        gt_mask: (H, W) 0/1 前景图

    Returns:
        dict，含 EBPG / IoU@20% / Hit@15% / BgRatio 四项；无前景时返回 None
    """
    h, w = gt_mask.shape
    cam_r = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    cam_r = np.clip(cam_r, 0, None)
    gt = (gt_mask > 0.5)

    if gt.sum() < 1:
        return None

    total = cam_r.sum()
    fg_energy = cam_r[gt].sum()

    # 1) EBPG：前景能量占比，无需阈值
    ebpg = float(fg_energy / total) if total > 1e-8 else 0.0

    # 2) IoU@k：取 CAM 最高的 k 比例像素作为预测前景
    def iou_at(ratio):
        k = max(1, int(cam_r.size * ratio))
        flat = cam_r.reshape(-1)
        thr = np.partition(flat, -k)[-k]
        pred = (cam_r >= thr)
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return float(inter / union) if union > 0 else 0.0

    # 3) Pointing Game：前 k 比例的高激活点落在 GT 内的比例
    def hit_at(ratio):
        k = max(1, int(cam_r.size * ratio))
        flat = cam_r.reshape(-1)
        idx = np.argpartition(flat, -k)[-k:]
        return float(gt.reshape(-1)[idx].mean())

    # 4) 背景抑制：背景平均激活 / 前景平均激活
    fg_mean = float(cam_r[gt].mean())
    bg_mean = float(cam_r[~gt].mean()) if (~gt).sum() > 0 else 0.0
    bg_ratio = float(bg_mean / fg_mean) if fg_mean > 1e-8 else float("inf")

    return {
        "EBPG": ebpg,
        "IoU@20%": iou_at(0.20),
        "Hit@15%": hit_at(0.15),
        "BgRatio": bg_ratio,
    }


# =========================================================================== #
# CAM 算法
# =========================================================================== #
def grad_cam_pp(activation, gradient):
    """Grad-CAM++。

    相比 Grad-CAM，用二阶/三阶梯度的比值给每个空间位置单独算权重 alpha，
    因此**同类多实例**时每个实例都会被激活（Grad-CAM 往往只高亮其中一个），
    这对实例分割场景（一张图多株病害）尤其重要。

    Args:
        activation: (1, C, H, W)
        gradient:   (1, C, H, W)

    Returns:
        (H, W) numpy，归一化到 [0, 1]
    """
    a = activation.detach()
    g = gradient.detach()

    g2 = g.pow(2)
    g3 = g.pow(3)
    sum_a = a.sum(dim=(2, 3), keepdim=True)

    alpha = g2 / (2.0 * g2 + sum_a * g3 + 1e-7)
    weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(dim=1, keepdim=True))
    return _normalize(cam[0, 0])


def eigen_cam(activation):
    """Eigen-CAM：对激活做 SVD 取第一主成分。

    不需要梯度，因此不受梯度饱和/消失影响，结果稳定且快，适合快速判断
    "网络到底有没有响应"。缺点是与具体类别无关。
    """
    a = activation.detach()[0]                  # (C, H, W)
    c, h, w = a.shape
    flat = a.flatten(1)
    flat = flat - flat.mean(dim=1, keepdim=True)    # 去中心

    try:
        _, _, vt = torch.linalg.svd(flat, full_matrices=False)
        proj = vt[0]
    except Exception:
        proj = flat.mean(dim=0)                 # SVD 不收敛时退回通道平均

    return _normalize(proj.reshape(h, w).abs())


def _normalize(x):
    """归一化到 [0, 1]；全等时返回全 0，避免除零。"""
    x = x.float()
    lo, hi = x.min(), x.max()
    if (hi - lo).abs() < 1e-8:
        return torch.zeros_like(x).cpu().numpy()
    return ((x - lo) / (hi - lo)).cpu().numpy()


def overlay_heatmap(img, cam, alpha=0.5):
    """把 CAM 以 JET 色图叠加到原图上。"""
    h, w = img.shape[:2]
    cam_u8 = np.uint8(255 * np.clip(cv2.resize(cam, (w, h)), 0, 1))
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    out = alpha * img.astype(np.float32) + (1 - alpha) * heatmap.astype(np.float32)
    return out.astype(np.uint8)


# =========================================================================== #
# ModelAdapter：屏蔽不同 meta architecture 的差异
# =========================================================================== #
class ModelAdapter(object):
    """把不同 meta architecture 的差异收敛到统一接口。

    子类只需实现 ``_compute_logits``（拿到用于反传的分类得分张量），
    特征提取 / 推理 / bases 等公共逻辑由基类处理。
    """

    def __init__(self, model, cfg, name):
        self.model = model
        self.cfg = cfg
        self.name = name
        self._hook_out = None
        self._handle = None

    # -------- 特征提取 --------
    def compute_features(self, batched):
        """跑 backbone(+FPN)，返回 OrderedDict[str, Tensor]。"""
        image = batched["image"]
        images = ImageList.from_tensors(
            [image], getattr(self.model.backbone, "size_divisibility", 32))
        norm = getattr(self.model, "normalizer", None)
        x = norm(images.tensor) if norm is not None else images.tensor
        return self.model.backbone(x)

    # -------- 子类实现 --------
    def _compute_logits(self, features):
        """返回 list of (1, C, H, W) 分类 logits；两阶段模型返回 None。"""
        raise NotImplementedError

    def target_score(self, features, detections):
        """返回 (score_tensor, class_id, level_index)，反传目标。"""
        raise NotImplementedError

    # -------- 公共：推理 --------
    @torch.no_grad()
    def detect(self, batched):
        self.model.eval()
        out = self.model([batched])
        return out[0]["instances"]

    # -------- 公共：bases（仅 BlendMask 有）--------
    def get_bases(self, features):
        return None

    # -------- 公共：FPN 层名 --------
    def fpn_levels(self, features):
        return [k for k in features.keys()]

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class BlendMaskAdapter(ModelAdapter):
    """adet BlendMask / BlendMask2（本项目）。

    单阶段 dense head：head 对每个 FPN 层输出 logits，
    可直接在网格上按框中心定位目标类别得分。
    """

    def _compute_logits(self, features):
        # forward_head 返回 (logits, reg, ctr)；top_layer 是 attention 分支
        return self.model.proposal_generator.forward_head(
            features, self.model.top_layer)[0]

    def target_score(self, features, detections):
        logits = self._compute_logits(features)
        return _dense_target_score(logits, detections)

    def get_bases(self, features, differentiable=False):
        """返回 bases (1, K, Hb, Wb)。

        ``differentiable=True`` 时保留计算图，以便把 bases 的能量作为 CAM 目标
        反传到 backbone —— 这才是衡量 basis module 行为的正确方式。
        """
        if not hasattr(self.model, "basis_module"):
            return None
        if differentiable:
            out, _ = self.model.basis_module(features, None)
        else:
            with torch.no_grad():
                out, _ = self.model.basis_module(features, None)
        return out["bases"][0]      # (1, K, Hb, Wb)

    def target_score(self, features, detections, cam_target="cls"):
        if cam_target == "mask":
            # 用 bases 的整体激活能量作为目标：直接反映"basis module 认为
            # 哪些区域需要生成原型"，与 mask 分支的改动直接相关。
            bases = self.get_bases(features, differentiable=True)
            if bases is None:
                raise RuntimeError("该模型无 basis_module，cam-target=mask 不可用")
            return bases.abs().mean(), 0, 0
        return _dense_target_score(self._compute_logits(features), detections)


class RetinaNetAdapter(ModelAdapter):
    """detectron2 RetinaNet。结构与 BlendMask 同为单阶段 dense head。"""

    def _compute_logits(self, features):
        return self.model.head(features)[0]

    def target_score(self, features, detections, cam_target="cls"):
        if cam_target == "mask":
            raise RuntimeError("RetinaNet 无 mask 分支，cam-target=mask 不可用")
        return _dense_target_score(self._compute_logits(features), detections)


class RCnnAdapter(ModelAdapter):
    """detectron2 GeneralizedRCNN 系列（Mask R-CNN / Faster R-CNN / Cascade）。

    两阶段模型没有"每个像素一个预测"的 dense logits，分类得分产生在
    RoIHead 的 box_predictor 里（对 RoIAlign 后的固定尺寸特征分类）。

    这里 hook 最后一级 box_predictor，取所有 proposal 中最大的分类 logit 作为
    反传目标。梯度可经 RoIAlign 流回 FPN 特征，因此 CAM 反映的是
    "最终分类决策依赖于输入图像的哪些区域"。
    """

    def _register(self):
        if self._handle is not None:
            return
        roi = self.model.roi_heads
        # Cascade R-CNN 用 box_predictors 列表；标准 RCNN 用 box_predictor
        target = getattr(roi, "box_predictor", None)
        if target is None:
            preds = getattr(roi, "box_predictors", None)
            target = preds[-1] if preds else None
        if target is None:
            return

        def hook(module, inp, out):
            # box_predictor 返回 (cls_score, bbox_pred) 或 cls_score
            self._hook_out = out[0] if isinstance(out, (tuple, list)) else out

        self._handle = target.register_forward_hook(hook)

    def _compute_logits(self, features):
        return None      # 两阶段：不用 dense logits

    def target_score(self, features, detections, cam_target="cls"):
        if cam_target == "mask":
            raise RuntimeError(
                "两阶段模型请用 cam-target=cls；mask 分支的 CAM 需单独 hook mask head")
        self._register()
        # 触发一次完整前向（含 roi_heads）以填充 hook 输出
        images = ImageList.from_tensors(
            [self._last_image], getattr(self.model.backbone, "size_divisibility", 32))
        norm = getattr(self.model, "normalizer", None)
        images.tensor = norm(images.tensor) if norm is not None else images.tensor
        self.model.roi_heads(images, features, self._last_proposals)

        if self._hook_out is None:
            raise RuntimeError("未能从 box_predictor 捕获分类得分，请检查模型结构")
        scores = self._hook_out
        if isinstance(scores, (list, tuple)):
            scores = torch.cat([s.flatten() for s in scores])
        flat = scores.flatten()
        idx = int(flat.argmax())
        cls = idx % scores.shape[-1]
        return flat.max(), int(cls), 0

    def detect(self, batched):
        """两阶段前向需要额外保存 proposals，供 target_score 复用。"""
        self.model.eval()
        with torch.no_grad():
            image = batched["image"]
            self._last_image = image
            images = ImageList.from_tensors(
                [image], getattr(self.model.backbone, "size_divisibility", 32))
            norm = getattr(self.model, "normalizer", None)
            images.tensor = norm(images.tensor) if norm is not None else images.tensor
            self._last_features = self.model.backbone(images.tensor)
            proposals, _ = self.model.proposal_generator(
                images, self._last_features, None)
            self._last_proposals = proposals
            out = self.model([batched])
        return out[0]["instances"]


class GenericAdapter(ModelAdapter):
    """兜底 adapter：任意带 backbone 的 detectron2 模型。

    只保证 Eigen-CAM（不需要梯度）可用；Grad-CAM++ 因缺少分类得分目标而跳过。
    """

    def _compute_logits(self, features):
        return None

    def target_score(self, features, detections, cam_target="cls"):
        raise RuntimeError(
            "模型 {} 无法自动构造得分目标，Grad-CAM++ 不可用；"
            "请改用 --method eigencam 或 bases".format(type(self.model).__name__))


def _dense_target_score(logits, detections):
    """单阶段模型：用得分最高的检测框中心，在各层网格上取对应类别 logit 求和。

    梯度同时流经所有 FPN 层。若无检测结果，退化为全图最大响应，保证不崩。
    """
    if detections is not None and len(detections) > 0:
        best = int(detections.scores.argmax())
        cls = int(detections.pred_classes[best])
        box = detections.pred_boxes[best].tensor[0]
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        img_h, img_w = detections.image_size

        total, best_lvl, max_s = None, 0, -1e9
        for li, logit in enumerate(logits):
            _, _, hl, wl = logit.shape
            gy = int(torch.clamp(cy / img_h * hl, 0, hl - 1))
            gx = int(torch.clamp(cx / img_w * wl, 0, wl - 1))
            s = logit[0, cls, gy, gx]
            total = s if total is None else total + s
            if float(s) > max_s:
                max_s, best_lvl = float(s), li
        return total, cls, best_lvl

    best_lvl, best_val = 0, None
    for li, logit in enumerate(logits):
        v = logit[0].max()
        if best_val is None or v > best_val:
            best_val, best_lvl = v, li
    last = logits[best_lvl]
    cls = int(last[0].flatten(1).max(dim=1)[0].argmax())
    return last[0].max(), cls, best_lvl


def build_adapter(model, cfg, name):
    """按模型结构自动选择 adapter（duck typing，不依赖 META_ARCHITECTURE 名字）。"""
    cls_name = type(model).__name__
    if hasattr(model, "basis_module") or cls_name in ("BlendMask", "BlendMask2"):
        return BlendMaskAdapter(model, cfg, name)
    if cls_name == "RetinaNet" or (
            hasattr(model, "head") and not hasattr(model, "proposal_generator")):
        return RetinaNetAdapter(model, cfg, name)
    if hasattr(model, "roi_heads"):
        return RCnnAdapter(model, cfg, name)
    return GenericAdapter(model, cfg, name)


# =========================================================================== #
# 单模型的 CAM 计算
# =========================================================================== #
def compute_cam(adapter, batched, target_layer, method,
                cam_target="cls", want_grad=True):
    """对一张图计算指定层的 CAM。

    Args:
        cam_target: ``cls`` 用分类得分反传（反映分类头关注区域）；
            ``mask`` 用 bases 能量反传（仅 BlendMask，直接反映 basis module 行为）

    Returns:
        dict，含 gcam（可为 None）/ ecam / detections / cls / bases
    """
    detections = adapter.detect(batched)
    features = adapter.compute_features(batched)

    result = {"detections": detections, "cls": None,
              "gcam": None, "ecam": None, "bases": None}

    if method in ("bases", "all"):
        result["bases"] = adapter.get_bases(features)

    # Eigen-CAM 不需要梯度，先算（对 GenericAdapter 也能用）
    if method in ("eigencam", "all") and target_layer in features:
        result["ecam"] = eigen_cam(features[target_layer])

    if not want_grad or method not in ("gradcampp", "all"):
        return result

    if target_layer not in features:
        raise KeyError("特征层 {!r} 不存在，可用层: {}".format(
            target_layer, adapter.fpn_levels(features)))

    feat = features[target_layer]
    feat.retain_grad()                       # 非叶子节点，必须显式保留梯度
    score, cls, lvl = adapter.target_score(features, detections, cam_target)
    result["cls"] = cls
    adapter.model.zero_grad(set_to_none=True)
    score.backward(retain_graph=True)
    if feat.grad is None:
        raise RuntimeError("未捕获到梯度，请确认目标得分与特征层在同一计算图上")
    result["gcam"] = grad_cam_pp(feat, feat.grad)
    return result


def compute_layer_cams(adapter, batched, cam_target="cls"):
    """对 FPN/BiFPN 各层分别算 Grad-CAM++，用于观察多尺度分工。

    小目标主要在高层分辨率（p3），大目标在低分辨率层（p5~p7）。对比各层可以
    确认 FPN 的分工是否正常，也能看出某模块主要影响了哪个尺度。

    注意：每层都**重新做一次前向**。因为 ``retain_grad`` + ``retain_graph=True``
    下多次 backward 会让 grad 累加，必须每层用干净的计算图。
    """
    detections = adapter.detect(batched)
    with torch.no_grad():
        levels = adapter.fpn_levels(adapter.compute_features(batched))

    out = {}
    for lname in levels:
        try:
            features = adapter.compute_features(batched)
            if lname not in features:
                continue
            feat = features[lname]
            feat.retain_grad()
            score, _, _ = adapter.target_score(features, detections, cam_target)
            adapter.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=True)
            if feat.grad is None:
                continue
            out[lname] = grad_cam_pp(feat, feat.grad)
            del features, feat
        except Exception as e:
            logger.warning("层 {} 计算 CAM 失败: {}: {}".format(
                lname, type(e).__name__, e))
    return out


def compute_layer_cams_safe(adapter, batched, cam_target, fallback="cls"):
    """各层 CAM；若 mask 目标不可用则自动退回 cls。"""
    try:
        return compute_layer_cams(adapter, batched, cam_target)
    except RuntimeError:
        if cam_target != fallback:
            logger.warning("cam-target={} 不可用，各层对比改用 {}".format(
                cam_target, fallback))
            return compute_layer_cams(adapter, batched, fallback)
        raise


# =========================================================================== #
# 数据集注册与 GT 查找
# =========================================================================== #
def set_thing_classes_from_json(name, json_file):
    """从 COCO json 设置类别名与 id 映射。

    ``register_coco_instances`` 对 ``thing_classes`` 是**懒设置**的：只有真正调用
    ``DatasetCatalog.get(name)`` 触发 ``load_coco_json`` 时才会写入 metadata。
    这里直接读 json，避免为拿类别名而加载整份数据。

    COCO 的 category id 常常不连续，而模型输出的 ``pred_classes`` 是 0-based 连续 id，
    因此必须给出映射关系，否则显示的类别名会错位。
    """
    with open(json_file, "r", encoding="utf-8") as f:
        cats = json.load(f).get("categories", [])
    cats = sorted(cats, key=lambda c: c["id"])
    if not cats:
        return
    MetadataCatalog.get(name).set(
        thing_classes=[c["name"] for c in cats],
        thing_dataset_id_to_contiguous_id={c["id"]: i for i, c in enumerate(cats)},
    )


def register_datasets(root):
    """扫描 datasets/ 下所有符合 COCO 结构的目录并注册。"""
    registered = []
    if not os.path.isdir(root):
        return registered
    for d in sorted(os.listdir(root)):
        dpath = os.path.join(root, d)
        if d.startswith(".") or not os.path.isdir(dpath):
            continue
        for split in ("train", "val"):
            js = os.path.join(dpath, "annotations", "instances_{}2017.json".format(split))
            imgdir = os.path.join(dpath, "{}2017".format(split))
            name = "{}_{}".format(d, split)
            if not (os.path.isfile(js) and os.path.isdir(imgdir)):
                continue
            if name in DatasetCatalog.list():
                continue
            register_coco_instances(name, {}, js, imgdir)
            set_thing_classes_from_json(name, js)
            registered.append(name)
    return registered


def datasets_root():
    env = os.environ.get("GBADMASK_DATA_ROOT")
    if env:
        parent = os.path.dirname(os.path.abspath(env))
        if os.path.isdir(parent):
            return parent
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def build_gt_index(ds_name):
    """建立 file_name -> (height, width, [anns]) 索引，用于取 GT 算指标。

    数据集未就绪时（例如 detectron2 自带 config 默认指向 coco_2017_val，而本机
    没有 datasets/coco）返回空字典而不是抛异常 —— 没有 GT 只是画不出对照图和
    算不了指标，CAM 本身仍应能正常输出。
    """
    if not ds_name or ds_name not in DatasetCatalog.list():
        return {}
    index = {}
    try:
        for rec in DatasetCatalog.get(ds_name):
            index[os.path.basename(rec["file_name"])] = (
                rec.get("height"), rec.get("width"), rec.get("annotations", []))
    except Exception as e:
        logger.warning("读取数据集 {} 失败（将无 GT 对照）: {}: {}".format(
            ds_name, type(e).__name__, e))
        return {}
    return index


# =========================================================================== #
# 绘图
# =========================================================================== #
def _drawn(img, detections, metadata):
    vis = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    return vis.draw_instance_predictions(detections.to("cpu")).get_image()[:, :, ::-1]


def _drawn_gt(img, gt_anns, metadata, alpha=0.45):
    """把 GT 前景以半透明绿色叠加，并画红色外接框，用于与 CAM 对照。

    这里不用 ``Visualizer``（它面向 ``Instances``/``dataset_dict`` 两种不同接口，
    GT 走 dataset_dict 路径还要额外处理 bbox_mode），直接画更可控且不依赖
    detectron2 内部细节。
    """
    if not gt_anns:
        return img
    h, w = img.shape[:2]
    mask = _gt_mask_from_anns(gt_anns, h, w)
    if mask.sum() < 1:
        return img

    out = img.astype(np.float32)
    color = np.array([80, 220, 120], dtype=np.float32)   # 绿
    fg = mask > 0.5
    out[fg] = out[fg] * (1 - alpha) + color * alpha

    out_bgr = out.astype(np.uint8)[:, :, ::-1].copy()
    for a in gt_anns:
        bbox = a.get("bbox")
        if not bbox:
            continue
        x, y, bw, bh = bbox
        cv2.rectangle(out_bgr, (int(x), int(y)),
                      (int(x + bw), int(y + bh)), (0, 0, 255), 2)
    return out_bgr[:, :, ::-1]


def save_single_overview(img, gt_anns, res, metadata, out_path, tag=""):
    panels, titles = [], []

    panels.append(_drawn_gt(img, gt_anns, metadata) if gt_anns else img)
    titles.append("Input + GT" if gt_anns else "Input")

    panels.append(_drawn(img, res["detections"], metadata))
    titles.append("{} dets ({})".format(tag, len(res["detections"])))

    if res["gcam"] is not None:
        panels.append(overlay_heatmap(img, res["gcam"]))
        titles.append("Grad-CAM++")
    if res["ecam"] is not None:
        panels.append(overlay_heatmap(img, res["ecam"]))
        titles.append("Eigen-CAM")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0))
    for ax, p, t in zip(np.atleast_1d(axes), panels, titles):
        ax.imshow(p)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_bases(bases, out_path):
    """画 basis module 输出的各张原型掩膜。"""
    b = bases[0].detach().float()
    k = b.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.6))
    for i, ax in enumerate(np.atleast_1d(axes)):
        bi = b[i]
        # 用 min-max 而不是 sigmoid：sigmoid 会把大值压平，看不出 bases 内部结构
        lo, hi = bi.min(), bi.max()
        arr = ((bi - lo) / (hi - lo + 1e-8)).cpu().numpy()
        ax.imshow(arr, cmap="magma")
        ax.set_title("base #{}".format(i), fontsize=10)
        ax.axis("off")
    fig.suptitle("Basis prototypes (K={})".format(k), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_layers(img, layer_cams, out_path, tag=""):
    """画各 FPN 层的 CAM 对比。"""
    if not layer_cams:
        return
    names = list(layer_cams.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.4))
    for ax, name in zip(np.atleast_1d(axes), names):
        cam = layer_cams[name]
        ax.imshow(overlay_heatmap(img, cam))
        ax.set_title("{}  (1/{})".format(name, cam.shape[0]), fontsize=11)
        ax.axis("off")
    fig.suptitle("Grad-CAM++ across levels {}".format(tag), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_comparison(img, gt_anns, results, metrics, metadata, out_path):
    """多模型并排：每行一个模型，列 = [GT] [Grad-CAM++] [叠加]。

    ``results`` / ``metrics`` 均为 {model_name: ...} 字典。
    """
    names = list(results.keys())
    nrow = len(names)
    fig, axes = plt.subplots(nrow, 3, figsize=(14, 4.8 * nrow))
    axes = np.atleast_2d(axes)

    for r, name in enumerate(names):
        res = results[name]
        m = (metrics or {}).get(name)
        gcam = res["gcam"]

        ax = axes[r, 0]
        ax.imshow(_drawn_gt(img, gt_anns, metadata) if gt_anns else img)
        ax.set_title("{}  |  dets={}".format(name, len(res["detections"])),
                     fontsize=12, loc="left")
        ax.axis("off")

        ax = axes[r, 1]
        if gcam is not None:
            ax.imshow(np.uint8(255 * cv2.resize(gcam, (img.shape[1], img.shape[0]))),
                      cmap="jet")
        ax.set_title("Grad-CAM++ (raw)", fontsize=11)
        ax.axis("off")

        ax = axes[r, 2]
        if gcam is not None:
            ax.imshow(overlay_heatmap(img, gcam))
            txt = ""
            if m:
                txt = "EBPG={:.3f}  IoU@20%={:.3f}  Hit@15%={:.3f}  BgRatio={:.3f}".format(
                    m["EBPG"], m["IoU@20%"], m["Hit@15%"], m["BgRatio"])
            ax.set_title(txt, fontsize=10)
        ax.axis("off")

    fig.suptitle("CAM comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bars(summary, out_path):
    """按模型汇总各指标画柱状图。"""
    if not summary:
        return
    keys = [k for k in ("EBPG", "IoU@20%", "Hit@15%", "BgRatio")
            if any(k in v for v in summary.values())]
    if not keys:
        return
    names = list(summary.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(names), 2)))
    for i, k in enumerate(keys):
        ax = np.atleast_1d(axes)[i]
        vals = [summary[nm].get(k, 0.0) for nm in names]
        bars = ax.bar(range(len(names)), vals, color=colors[:len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_title("{} (avg)".format(k), fontsize=11)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, "{:.3f}".format(v),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# 主流程
# =========================================================================== #
def load_model(cfg_path, weights, opts=None, device="cuda"):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    model = build_model(cfg)
    model.eval()
    if weights and os.path.isfile(weights):
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            weights, resume=False)
    elif weights:
        logger.warning("权重文件不存在，使用随机初始化: {}".format(weights))
    return model.to(device), cfg


def process(models, paths, gt_index, metadata, args):
    """models: {name: (adapter, cfg)}"""
    all_rows = []
    summary = {name: {} for name in models}

    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        img = utils.read_image(p, format="RGB")
        h, w = img.shape[:2]
        batched = {"image": torch.as_tensor(
            img.astype("float32").transpose(2, 0, 1)).to(args.device),
            "height": h, "width": w}

        gt_anns = gt_index.get(os.path.basename(p), (None, None, []))[2]
        gt_mask = None
        if args.metric and gt_anns:
            gt_mask = _gt_mask_from_anns(gt_anns, h, w)

        results, metrics = {}, {}
        for name, (adapter, cfg) in models.items():
            try:
                res = compute_cam(adapter, batched, args.target_layer,
                                  args.method, args.cam_target)
                results[name] = res
                if gt_mask is not None and res["gcam"] is not None:
                    m = compute_metrics(res["gcam"], gt_mask)
                    if m:
                        metrics[name] = m
                        all_rows.append(dict(image=stem, model=name, **m))
                logger.info("  [{}] {} -> cls {}, {} dets".format(
                    name, stem, res["cls"], len(res["detections"])))
            except Exception as e:
                logger.error("  [{}] {} 失败: {}: {}".format(
                    name, stem, type(e).__name__, e))
                import traceback
                traceback.print_exc()

        if not results:
            continue

        # 输出
        if len(models) == 1:
            name = list(results)[0]
            save_single_overview(img, gt_anns, results[name], metadata,
                                 os.path.join(args.output_dir,
                                              "{}_overview.jpg".format(stem)), name)
        else:
            save_comparison(img, gt_anns, results, metrics, metadata,
                            os.path.join(args.output_dir, "{}_cmp.jpg".format(stem)))

        # bases 图：method 为 bases/all 时已取到，直接保存（无需额外开关）
        for name, res in results.items():
            if res["bases"] is not None:
                save_bases(res["bases"], os.path.join(
                    args.output_dir, "{}_{}_bases.jpg".format(stem, name)))

        # 各 FPN 层的 CAM 对比
        if args.all_layers:
            for name, (adapter, cfg) in models.items():
                try:
                    lcs = compute_layer_cams_safe(adapter, batched, args.cam_target)
                    if lcs:
                        save_layers(img, lcs, os.path.join(
                            args.output_dir, "{}_{}_layers.jpg".format(stem, name)),
                            "({})".format(name))
                except Exception as e:
                    logger.warning("  [{}] 各层 CAM 失败: {}".format(name, e))

    # 汇总
    if all_rows:
        for name in summary:
            rows = [r for r in all_rows if r["model"] == name]
            if not rows:
                continue
            for k in ("EBPG", "IoU@20%", "Hit@15%", "BgRatio"):
                vals = [r[k] for r in rows if np.isfinite(r[k])]
                summary[name][k] = float(np.mean(vals)) if vals else 0.0
    return all_rows, summary


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(
        description="Visualize and compare CAM heatmaps across models",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", metavar="FILE",
                        help="单模型模式的配置文件（与 --models 二选一）")
    parser.add_argument("--models", nargs="+", metavar="NAME:CONFIG:WEIGHTS",
                        help="多模型对比，例如 baseline:cfg.yaml:model.pth")
    parser.add_argument("--input", nargs="+", required=True,
                        help="图片路径，可多个，每项支持 glob")
    parser.add_argument("--output-dir", default="output/cam")
    parser.add_argument("--method", default="all",
                        choices=["gradcampp", "eigencam", "bases", "all"])
    parser.add_argument("--target-layer", default="p3",
                        help="CAM 目标层：p3~p7 之一（默认 p3）")
    parser.add_argument("--cam-target", default="cls", choices=["cls", "mask"],
                        help="反传目标：cls=分类得分（看分类头关注哪里）；"
                             "mask=bases 能量（看 basis module 关注哪里，"
                             "评估 mask 分支改动时应用这个，仅 BlendMask 支持）")
    parser.add_argument("--all-layers", action="store_true",
                        help="额外输出 p3~p7 各层的 CAM 对比（每层单独前向，较慢）")
    parser.add_argument("--metric", action="store_true",
                        help="计算定量指标（需要图片属于已注册的数据集）")
    parser.add_argument("--dataset", default=None,
                        help="数据集名（取 GT 与类别名）；默认取 cfg.DATASETS.TEST[0]")
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="覆盖配置，例如 MODEL.WEIGHTS path/to/model.pth")
    return parser.parse_args(in_args)


def main(in_args=None):
    args = parse_args(in_args)
    os.makedirs(args.output_dir, exist_ok=True)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 组装待比较的模型列表
    specs = []
    if args.models:
        for item in args.models:
            parts = item.split(":")
            if len(parts) != 3:
                sys.exit("--models 每项格式须为 NAME:CONFIG:WEIGHTS，收到: " + item)
            specs.append(tuple(parts))
    elif args.config_file:
        w = ""
        if args.opts and "MODEL.WEIGHTS" in args.opts:
            i = args.opts.index("MODEL.WEIGHTS")
            if i + 1 < len(args.opts):
                w = args.opts[i + 1]
                args.opts = args.opts[:i] + args.opts[i + 2:]
        specs.append(("model", args.config_file, w))
    else:
        sys.exit("请用 --config-file 或 --models 指定模型")

    # 2) 注册数据集（GT 与类别名来源）
    newly = register_datasets(datasets_root())
    if newly:
        logger.info("已注册数据集: {}".format(", ".join(newly)))

    # 3) 加载模型
    models = {}
    metadata = None
    for name, cfg_path, weights in specs:
        model, cfg = load_model(cfg_path, weights, args.opts, args.device)
        adapter = build_adapter(model, cfg, name)
        models[name] = (adapter, cfg)
        logger.info("模型 {}: {} (adapter={})".format(
            name, type(model).__name__, type(adapter).__name__))

        if metadata is None:
            ds = args.dataset or (cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else None)
            if ds:
                try:
                    md = MetadataCatalog.get(ds)
                    if md.thing_classes:
                        metadata = md
                except Exception:
                    pass
    if metadata is None:
        from detectron2.data import Metadata
        metadata = Metadata()
        metadata.set(thing_classes=[])
        logger.warning("未能获取类别名，检测结果将不显示类别标签")

    # 4) GT 索引
    gt_index = {}
    if args.metric or True:      # GT 也用于可视化，默认构建
        ds = args.dataset
        if not ds:
            for _, (_, c) in models.items():
                if c.DATASETS.TEST:
                    ds = c.DATASETS.TEST[0]
                    break
        gt_index = build_gt_index(ds)
        if args.metric and not gt_index:
            logger.warning("未找到 GT 标注，--metric 将不生效")

    # 5) 图片列表
    paths = []
    for p in args.input:
        matched = sorted(glob.glob(p))
        paths.extend(matched if matched else [p])
    paths = sorted(set(paths))
    logger.info("待处理 {} 张图，模型: {}".format(len(paths), list(models)))

    # 6) 跑
    rows, summary = process(models, paths, gt_index, metadata, args)

    # 7) 写结果
    if rows:
        csv_path = os.path.join(args.output_dir, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info("逐图指标: {}".format(csv_path))

    if summary and any(summary.values()):
        txt_path = os.path.join(args.output_dir, "metrics_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("{:<20s} {:>10s} {:>10s} {:>10s} {:>10s}\n".format(
                "model", "EBPG", "IoU@20%", "Hit@15%", "BgRatio"))
            for name, m in summary.items():
                if not m:
                    continue
                f.write("{:<20s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(
                    name, m["EBPG"], m["IoU@20%"], m["Hit@15%"], m["BgRatio"]))
        logger.info("汇总指标:\n" + open(txt_path, encoding="utf-8").read())
        save_metrics_bars({k: v for k, v in summary.items() if v},
                          os.path.join(args.output_dir, "metrics_summary.jpg"))

    logger.info("完成，结果在 {}".format(args.output_dir))


if __name__ == "__main__":
    main()
