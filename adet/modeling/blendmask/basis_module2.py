from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform

from .basis_module import BASIS_MODULE_REGISTRY, build_attention
from .fdc_loss import DC_and_CE_loss

__all__ = ["ProtoNetV2", "build_basis_module2"]


def build_basis_module2(cfg, input_shape):
    """向后兼容入口。

    两份 ProtoNet 现在注册在同一个 registry（``BASIS_MODULE``）里，
    通过 ``MODEL.BASIS_MODULE.NAME`` 选择，因此本函数与
    :func:`basis_module.build_basis_module` 等价，仅为不破坏旧调用而保留。
    """
    return BASIS_MODULE_REGISTRY.get(cfg.MODEL.BASIS_MODULE.NAME)(cfg, input_shape)


@BASIS_MODULE_REGISTRY.register()
class ProtoNetV2(nn.Module):
    """改进版 ProtoNet（GBADMask 的核心改动之一）。

    相对官方 :class:`basis_module.ProtoNet` 的四处改动：

    1. **低层细节分支**：取 ``in_features[0]``（最高分辨率层）经 1×1 卷积降到
       ``LOW_LEVEL_DIM`` 通道，保留边缘/纹理等细节；
    2. **注意力增强**：低层分支与 tower 开头各插入一个注意力模块，
       由 ``MODEL.BASIS_MODULE.ATTN`` 在 gc / cbam / ca / none 之间切换；
    3. **深浅特征融合**：低层特征上采样后与 tower 输出在通道维 concat，
       再过一层 3×3 卷积降回 ``CONVS_DIM``；
    4. **bases 输出头**：tower 尾部的 1×1 被移除，改由 ``self.conv2``
       （3×3-BN-ReLU-1×1）输出，多一层非线性。

    另外，辅助语义分割损失由官方的 ``F.cross_entropy`` 换成
    :class:`fdc_loss.DC_and_CE_loss`（Focal-Dice-CE）。
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        mask_dim       = cfg.MODEL.BASIS_MODULE.NUM_BASES
        planes         = cfg.MODEL.BASIS_MODULE.CONVS_DIM
        self.in_features = cfg.MODEL.BASIS_MODULE.IN_FEATURES
        self.loss_on   = cfg.MODEL.BASIS_MODULE.LOSS_ON
        norm           = cfg.MODEL.BASIS_MODULE.NORM
        num_convs      = cfg.MODEL.BASIS_MODULE.NUM_CONVS
        self.visualize = cfg.MODEL.BLENDMASK.VISUALIZE
        attn           = cfg.MODEL.BASIS_MODULE.ATTN
        low_dim        = cfg.MODEL.BASIS_MODULE.LOW_LEVEL_DIM
        # fmt: on

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        # 低层细节分支：1×1 卷积降维，padding=0（原实现误用 padding=1，
        # 对 1×1 卷积无意义且会使空间尺寸 +2，靠后续插值掩盖）
        self.conv1 = nn.Conv2d(
            feature_channels[self.in_features[0]], low_dim,
            kernel_size=1, stride=1, padding=0, bias=False)

        self.attn_low = build_attention(attn, low_dim)
        self.attn_tower = build_attention(attn, planes)

        self.dc_ce_loss = DC_and_CE_loss()

        conv_block = conv_with_kaiming_uniform(norm, True)  # conv relu bn
        self.concat = conv_block(planes + low_dim, planes, 3, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, mask_dim, 1))

        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature], planes, 3, 1))

        tower = [self.attn_tower]
        for i in range(num_convs):
            tower.append(conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower.append(conv_block(planes, planes, 3, 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.loss_on:
            # fmt: off
            self.common_stride   = cfg.MODEL.BASIS_MODULE.COMMON_STRIDE
            num_classes          = cfg.MODEL.BASIS_MODULE.NUM_CLASSES + 1
            self.sem_loss_weight = cfg.MODEL.BASIS_MODULE.LOSS_WEIGHT
            # fmt: on

            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(),
                nn.Conv2d(planes, num_classes, kernel_size=1, stride=1))

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                x_p = F.interpolate(
                    x_p, x.size()[2:], mode="bilinear", align_corners=True)
                x = x + x_p

        fre = self.tower(x)

        low_feat = self.conv1(features[self.in_features[0]])
        low_feat = F.interpolate(
            low_feat, fre.size()[2:], mode="bilinear", align_corners=True)

        # 注意力加权的低层细节 + 深层语义，concat 后降维
        x = torch.cat((self.attn_low(low_feat), fre), 1)
        x = self.concat(x)
        outputs = {"bases": [self.conv2(x)]}

        losses = {}
        if self.training and self.loss_on:
            sem_out = self.seg_head(features[self.in_features[0]])
            # resize target to reduce memory
            gt_sem = targets.unsqueeze(1).float()
            gt_sem = F.interpolate(gt_sem, scale_factor=1 / self.common_stride)
            seg_loss = self.dc_ce_loss(sem_out, gt_sem.squeeze(1).long())
            losses['loss_basis_sem'] = seg_loss * self.sem_loss_weight
        elif self.visualize and hasattr(self, "seg_head"):
            outputs["seg_thing_out"] = self.seg_head(features[self.in_features[0]])
        return outputs, losses
