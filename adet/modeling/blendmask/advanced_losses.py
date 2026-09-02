# -*- coding: utf-8 -*-
"""面向前景稀少的实例/语义分割损失。

背景
----
本项目原用 ``fdc_loss.DC_and_CE_loss``（CE + SoftDice 的加权和，外再套一层 focal
调制）。实测在小麦病害数据集上**严重损害 mask 分支**（segm AP 6.797 → 0.611），
排查后根因不是损失公式，而是：

> 语义分割中背景像素占绝大多数，模型倾向把 p3 特征**平滑化**以利于预测大片背景；
> 而 BlendMask 的 bases 恰恰依赖这些高频细节。检测头对细节不敏感故 bbox 正常，
> bases 受损故 segm 崩溃。

因此本模块提供两类改进：

1. **对前景更友好的损失**：Focal Tversky、Unified Focal。
   Tversky 是 Dice 的推广，可用 α/β 分别控制 FP / FN 的惩罚权重——
   前景稀少时把 β 调大（更惩罚漏检）能显著提升召回。
2. **梯度截断（detach）**：见 :func:`BasisSemHeadWrapper`，
   让语义监督只训练 head，不回传污染共享的 backbone 特征。

参考文献
--------
* Focal Tversky Loss: Abraham & Khan, *A Novel Focal Tversky Loss Function With
  Improved Attention U-Net for Lesion Segmentation*, ISBI 2019
  （2024–2025 的医学/遥感小目标分割工作中仍被广泛采用）
* Unified Focal Loss: Yeung et al., *Unified Focal Loss: Generalising Dice and
  Cross Entropy-based Losses to Handle Class Imbalanced Medical Image
  Segmentation*, Computerized Medical Imaging and Graphics 2022
* Asymmetric Loss: Ben-Baruch et al., ICCV 2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target, num_classes):
    """(B, H, W) int64 -> (B, C, H, W) float。"""
    return F.one_hot(target.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()


class TverskyLoss(nn.Module):
    """Tversky loss（Dice 的推广）。

    Args:
        alpha: FP 的惩罚权重
        beta:  FN 的惩罚权重。``alpha + beta = 1`` 时退化为 Dice 的形式；
               **前景稀少时设 ``beta > alpha``**，更惩罚漏检，提升召回。
        smooth: 数值稳定项
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, do_bg=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        y = _one_hot(target, num_classes)

        start = 0 if self.do_bg else 1
        probs = probs[:, start:]
        y = y[:, start:]

        tp = (probs * y).sum(dim=(2, 3))
        fp = (probs * (1 - y)).sum(dim=(2, 3))
        fn = ((1 - probs) * y).sum(dim=(2, 3))

        ti = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - ti).mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss。

    在 Tversky loss 上加 focal 调制 ``(1 - TI)^(1/gamma)``，把训练重心压向难样本。

    Args:
        gamma: 调制指数，常用 4/3 ~ 3。值越大对易样本抑制越强。
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=4.0 / 3.0,
                 smooth=1e-6, do_bg=True):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth, do_bg)
        self.gamma = gamma

    def forward(self, logits, target):
        ti_loss = self.tversky(logits, target)      # = 1 - TI 的均值
        return (ti_loss + 1e-8) ** (1.0 / self.gamma)


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal loss（ICCV 2021），对正负样本用不同的 gamma。

    正负样本极不平衡时，单一边界的 focal 会同时压低正样本的梯度；
    拆成正负两个 gamma 后可分别控制。
    """

    def __init__(self, gamma_pos=1.0, gamma_neg=3.0, clip=0.05, do_bg=True):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.do_bg = do_bg

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        y = _one_hot(target, num_classes)
        start = 0 if self.do_bg else 1
        probs = probs[:, start:]
        y = y[:, start:]

        # 概率裁剪，避免易负样本梯度爆炸
        if self.clip is not None and self.clip > 0:
            probs = (probs * (1 - 2 * self.clip) + self.clip).clamp(0, 1)

        pt = torch.where(y == 1, probs, 1 - probs)
        gamma = torch.where(y == 1, self.gamma_pos, self.gamma_neg)
        loss = -((1 - pt) ** gamma) * torch.log(pt.clamp(min=1e-8))
        return loss.mean()


class UnifiedFocalLoss(nn.Module):
    """Unified Focal loss（2022）。

    统一了 Dice 系与 CE 系两类损失，用两个超参分别控制：

    * ``delta``：Dice 项（Tversky）与 CE 项的配比
    * ``gamma``：focal 调制强度（同时作用于两项）

    ``delta=0`` 退化为纯 Focal CE；``delta=1`` 退化为纯 Focal Tversky。
    """

    def __init__(self, weight=0.5, delta=0.6, gamma=0.75,
                 alpha=0.3, beta=0.7, do_bg=True):
        super().__init__()
        self.weight = weight          # 兼容性保留（语义损失总权重另有 LOSS_WEIGHT）
        self.delta = delta
        self.gamma = gamma
        self.ftl = FocalTverskyLoss(alpha, beta, gamma=gamma, do_bg=do_bg)
        self.afl = AsymmetricFocalLoss(gamma_pos=gamma, gamma_neg=gamma * 2,
                                       do_bg=do_bg)

    def forward(self, logits, target):
        ftl = self.ftl(logits, target)
        afl = self.afl(logits, target)
        return self.delta * ftl + (1 - self.delta) * afl


def build_sem_loss(name, do_bg=False):
    """按名字构造语义损失。

    Args:
        name: ``fdc`` / ``focal_tversky`` / ``unified_focal`` / ``ce``
        do_bg: 是否把背景计入损失项。前景稀少时建议 ``False``，
               否则背景（几乎总能预测对）会主导损失、使梯度趋近于 0。
    """
    name = (name or "ce").lower()
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=-100)
    if name == "fdc":
        from .fdc_loss import DC_and_CE_loss
        return DC_and_CE_loss(soft_dice_kwargs={"do_bg": do_bg})
    if name == "focal_tversky":
        return FocalTverskyLoss(alpha=0.3, beta=0.7, do_bg=do_bg)
    if name == "unified_focal":
        return UnifiedFocalLoss(delta=0.6, gamma=0.75, do_bg=do_bg)
    raise ValueError("未知的语义损失: {!r}，可选 ce/fdc/focal_tversky/unified_focal"
                     .format(name))


class BasisSemHeadWrapper(nn.Module):
    """语义辅助头的包装，支持**梯度截断**。

    原实现中 ``seg_head`` 直接吃 backbone 特征，语义损失的梯度会回传 backbone。
    由于语义分割倾向平滑化特征（背景占绝大多数），这会损害 bases 所需的高频细节。

    ``detach=True`` 时把输入特征 ``.detach()``，语义损失**只训练 seg_head 本身**，
    不再污染共享特征。这样既保留了语义分支对 bases 的正则作用（若有），
    又避免了特征退化。

    注意：开启 detach 后语义 head 变成纯粹的辅助分支，对 bases 无正则作用；
    是否仍有收益需实测确认。
    """

    def __init__(self, head, detach=True):
        super().__init__()
        self.head = head
        self.detach = detach

    def forward(self, features):
        x = features.detach() if self.detach else features
        return self.head(x)
