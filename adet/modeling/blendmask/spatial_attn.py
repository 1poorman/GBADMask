"""空间注意力模块。

为什么单独提供纯空间注意力
--------------------------
bases 的核心任务是**空间定位**：哪些像素是前景、边界在哪里。

而常用的 GCNet / SE 属于**通道注意力**，其权重 shape 为 (B, C, 1, 1)，
对空间广播 —— 它回答的是"哪些通道重要"，而不是"哪些位置重要"。
本项目 :class:`adet.modeling.blendmask.GCblock.GlobalContextBlock` 默认用的
正是空间无关的 ``channel_mul`` 融合方式。

把 CBAM 里的 spatial 分支单独抽出来，就可以与 GC 做对照实验：
若纯空间注意力优于纯通道注意力，即可验证"bases 需要空间定位能力"这一判断。

配合 ``MODEL.BASIS_MODULE.ATTN`` 使用，可选值见
:func:`adet.modeling.blendmask.basis_module.build_attention`。
"""
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力分支（不含通道注意力）。

    沿通道维做 max / avg 得到两张 (B,1,H,W) 描述子，拼接后用一层卷积得到
    (B,1,H,W) 的空间权重图，sigmoid 后与输入相乘。
    """

    def __init__(self, channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return x * attn
