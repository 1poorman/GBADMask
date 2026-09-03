# Copyright (c) GBADMask. All Rights Reserved.
"""
EMA —— Efficient Multi-Scale Attention（ICASSP 2023，跨空间学习）

设计来源
--------
- 论文 "Efficient Multi-Scale Attention Module with Cross-Spatial Learning"（ICASSP 2023）。
- 核心思想：在不做通道降维的前提下，用 1×1（跨空间）与 3×3（局部）两条并行的
  多尺度分支重塑通道权重，并通过全局上下文做跨空间交互。相比 CA，它显式引入
  3×3 局部分支，更利于保留病斑这类小目标的局部纹理。

实现说明
--------
- 严格复刻开源权威实现（CSDN/主流 repo 版本）：把通道分组后，对每组取水平/垂直
  1D 池化 + 全局池化，经 1×1 跨空间卷积与 3×3 局部分支做交叉注意力，最后以
  sigmoid 门控回乘原特征。输出通道数不变（满足 ``c1 == c2``）。
- ``factor`` 为分组基数（论文默认 32）。由于本仓库 basis 低层分支仅
  ``LOW_LEVEL_DIM=24`` 通道，``factor`` 会自动向下取一个能整除通道数、
  且每组通道数 ≥ 4 的值，避免在 24 通道上分组失败或退化。
- 可直接挂到 ``basis_module`` 的 ``attn_low`` / ``attn_tower`` 位置：
  ``MODEL.BASIS_MODULE.ATTN = "ema"``（与 gc / psa 同接口）。
"""
import torch
from torch import nn


def _ema_groups(c: int, factor: int) -> int:
    """选定能整除 c、且每组通道数 ≥ 4 的最大分组基数（≤ factor）。"""
    cap = max(1, min(factor, c // 4))
    g = cap
    while g > 1 and c % g != 0:
        g -= 1
    return max(1, g)


class EMA(nn.Module):
    def __init__(self, channels: int, factor: int = 32):
        super().__init__()
        self.groups = _ema_groups(channels, factor)
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        cg = channels // self.groups
        self.gn = nn.GroupNorm(cg, cg)
        # 裸卷积（无 BN/ReLU）——参考实现语义。若加 ReLU，门控 sigmoid 的输入
        # 非负，门值被限制在 [0.5, 1]，注意力只能放大不能抑制，行为实质改变
        self.conv1x1 = nn.Conv2d(cg, cg, 1)
        self.conv3x3 = nn.Conv2d(cg, cg, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # (b*g, c//g, h, w)
        x_h = self.pool_h(group_x)                       # (b*g, c//g, 1, w)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)    # (b*g, c//g, 1, h)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))   # (b*g, c//g, 1, w+h)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # 跨空间门控：水平/垂直 sigmoid 相乘，再经 GroupNorm
        x1 = self.gn(
            group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        )
        x2 = self.conv3x3(group_x)
        # 交叉空间注意力：全局上下文对两分支做软分配
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # (b*g, c//g, hw)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (
            torch.matmul(x11, x12) + torch.matmul(x21, x22)
        ).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
