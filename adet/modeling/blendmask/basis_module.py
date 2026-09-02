from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform

from .ca import CA_Block
from .cbam import CBAMLayer
from .GCblock import GlobalContextBlock
from .spatial_attn import SpatialAttention


BASIS_MODULE_REGISTRY = Registry("BASIS_MODULE")
BASIS_MODULE_REGISTRY.__doc__ = """
Registry for basis module, which produces global bases from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.

Registered names:
  ProtoNet   -- 官方实现
  ProtoNetV2 -- 本项目改进版（低层特征 + 注意力融合 + Focal-Dice-CE 损失）
"""


def build_attention(name, channels):
    """按名字构造注意力模块，用于 basis 模块内部的特征增强。

    支持的取值见 ``MODEL.BASIS_MODULE.ATTN``：

    ============ ==========================================================
    取值          含义
    ============ ==========================================================
    ``none``     ``nn.Identity()``，不加注意力
    ``gc``       GCNet GlobalContextBlock，**通道**注意力（空间无关）
    ``cbam``     CBAM，通道 + 空间注意力串联
    ``ca``       Coordinate Attention，沿 H/W 分别编码位置信息
    ``spatial``  **纯空间**注意力（CBAM 的 spatial 分支单独抽出）
    ============ ==========================================================

    ``gc`` 与 ``spatial`` 构成一组对照：前者只做通道重标定，后者只做空间
    重标定，可用于判断 bases 更需要哪一类能力。
    """
    if not name or name == "none":
        return nn.Identity()
    name = name.lower()
    if name == "gc":
        return GlobalContextBlock(channels, ratio=1 / 16)
    if name == "cbam":
        return CBAMLayer(channels)
    if name == "ca":
        return CA_Block(channels)
    if name == "spatial":
        return SpatialAttention(channels)
    raise ValueError(
        "未知的 MODEL.BASIS_MODULE.ATTN: {!r}，"
        "可选 none/gc/cbam/ca/spatial".format(name)
    )


class AddCoord(nn.Module):
    """:func:`add_coord_features` 的 Module 包装，便于放进 ``nn.Sequential``。"""

    def forward(self, x):
        return add_coord_features(x)


def add_coord_features(x):
    """在通道维拼接 2 张归一化坐标图（相对坐标编码）。

    BlendMask 的 bases 是**全图共享**、位置无关的原型，网络只能靠卷积的权值共享
    隐式感知位置，这对"哪些像素是前景"这一空间任务是天然不利的。

    CondInst 已验证：给 mask head 输入相对坐标图能显著提升实例掩膜质量。
    这里把同样的技巧用在 bases 生成上，代价仅为 2 个输入通道。

    Args:
        x: (B, C, H, W)

    Returns:
        (B, C+2, H, W)
    """
    b, _, h, w = x.shape
    device, dtype = x.device, x.dtype
    # 归一化到 [-1, 1]，与输入分辨率无关
    yy = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(1, 1, h, 1)
    xx = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, 1, 1, w)
    yy = yy.expand(b, 1, h, w)
    xx = xx.expand(b, 1, h, w)
    return torch.cat([x, xx, yy], dim=1)


def build_basis_module(cfg, input_shape):
    name = cfg.MODEL.BASIS_MODULE.NAME
    return BASIS_MODULE_REGISTRY.get(name)(cfg, input_shape)


@BASIS_MODULE_REGISTRY.register()
class ProtoNet(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        TODO: support deconv and variable channel width
        """
        # official protonet has a relu after each conv
        super().__init__()
        # fmt: off
        mask_dim          = cfg.MODEL.BASIS_MODULE.NUM_BASES
        planes            = cfg.MODEL.BASIS_MODULE.CONVS_DIM
        self.in_features  = cfg.MODEL.BASIS_MODULE.IN_FEATURES
        self.loss_on      = cfg.MODEL.BASIS_MODULE.LOSS_ON
        norm              = cfg.MODEL.BASIS_MODULE.NORM
        num_convs         = cfg.MODEL.BASIS_MODULE.NUM_CONVS
        self.visualize    = cfg.MODEL.BLENDMASK.VISUALIZE
        self.coord_on     = cfg.MODEL.BASIS_MODULE.COORD_ON
        # fmt: on

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, True)  # conv relu bn
        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature], planes, 3, 1))
        # 位置编码：tower 前拼接 2 通道坐标图，因此 tower 首层输入需 +2
        tower_in = planes + 2 if self.coord_on else planes
        tower = [conv_block(tower_in, planes, 3, 1)]
        for i in range(1, num_convs):
            tower.append(
                conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower.append(
            conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Conv2d(planes, mask_dim, 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.loss_on:
            # fmt: off
            self.common_stride   = cfg.MODEL.BASIS_MODULE.COMMON_STRIDE
            num_classes          = cfg.MODEL.BASIS_MODULE.NUM_CLASSES + 1
            self.sem_loss_weight = cfg.MODEL.BASIS_MODULE.LOSS_WEIGHT
            # fmt: on

            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, num_classes, kernel_size=1,
                                                    stride=1))

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                x_p = F.interpolate(x_p, x.size()[2:], mode="bilinear", align_corners=False)
                # x_p = aligned_bilinear(x_p, x.size(3) // x_p.size(3))
                x = x + x_p
        if self.coord_on:
            x = add_coord_features(x)
        outputs = {"bases": [self.tower(x)]}
        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.loss_on:
            sem_out = self.seg_head(features[self.in_features[0]])
            # resize target to reduce memory。
            # 原实现用 scale_factor=1/common_stride 降采样，但 backbone 实际输出的
            # 分辨率受 padding / size_divisibility 影响，未必等于 1/common_stride，
            # 配置与实际不符时会在 cross_entropy 处报尺寸不匹配。
            # 这里直接对齐到 sem_out 的空间尺寸，与 COMMON_STRIDE 解耦。
            gt_sem = targets.unsqueeze(1).float()
            gt_sem = F.interpolate(gt_sem, size=sem_out.shape[-2:],
                                   mode="nearest")
            seg_loss = F.cross_entropy(
                sem_out, gt_sem.squeeze(1).long())
            losses['loss_basis_sem'] = seg_loss * self.sem_loss_weight
        elif self.visualize and hasattr(self, "seg_head"):
            outputs["seg_thing_out"] = self.seg_head(features[self.in_features[0]])
        return outputs, losses
