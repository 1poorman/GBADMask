# -*- coding: utf-8 -*-
"""DySample：基于点采样的超轻量上采样算子（ICLR 2024）。

为什么用它替换 BiFPN 里的 nearest / bilinear
--------------------------------------------
特征金字塔的融合依赖上采样把低分辨率特征放大后再与各层相加。

* ``nearest``：直接复制像素，会在放大图上引入明显的"块状"错位，
  使得跨尺度相加时语义与空间位置对不齐；
* ``bilinear``：固定权重，无法根据内容自适应，边缘会被平滑；
* **DySample**：为每个目标位置学习一个**内容相关的采样偏移**，
  让上采样后的特征与同层特征在空间上对齐得更好，
  而且只由一个 ``1×1``（或 ``3×3``）卷积 + ``grid_sample`` 组成，
  参数量与计算量都极小。

原始 BiFPN（EfficientDet）只用了最简单的 resize，这正是它的改进空间之一。

参考
----
Liu et al., *Learning to Upsample by Learning to Sample*, ICLR 2024.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """DySample 上采样。

    Args:
        in_channels: 输入通道数
        scale: 放大倍数（2 表示尺寸翻倍）
        style: ``"lp"``（默认，静态+动态范围因子）或 ``"pl"``（纯动态）
        groups: 分组数，4 是论文默认值（在精度与开销间平衡）
    """

    def __init__(self, in_channels, scale=2, style="lp", groups=4,
                 dyscope=False):
        super().__init__()
        if scale != 2:
            raise ValueError("当前实现仅支持 scale=2（BiFPN 相邻层固定 2 倍）")
        self.scale = scale
        self.style = style
        self.groups = groups

        if in_channels < groups or in_channels % groups != 0:
            raise ValueError(
                "in_channels({}) 必须能被 groups({}) 整除".format(
                    in_channels, groups))

        if style == "pl":
            # 直接预测 2*g 个通道的偏移（x, y 各 g 组），取组平均后加在网格上。
            self.offset = nn.Conv2d(in_channels, 2 * groups, kernel_size=1)
            nn.init.constant_(self.offset.weight, 0)
            nn.init.constant_(self.offset.bias, 0)
            self.scope = None
        elif style == "lp":
            # 每个输出像素对应 scale×scale 组 (x, y)，每组再分 g 个子组
            n_off = 2 * groups * scale * scale
            self.offset = nn.Conv2d(in_channels, n_off, kernel_size=1)
            nn.init.constant_(self.offset.weight, 0)
            nn.init.constant_(self.offset.bias, 0)
            self.scope = nn.Conv2d(in_channels, n_off, kernel_size=1)
            nn.init.constant_(self.scope.weight, 0)
            nn.init.constant_(self.scope.bias, 0)
            self.register_buffer("init_pos", self._init_pos())
        else:
            raise ValueError("style 必须是 'lp' 或 'pl'，收到 " + str(style))

        # 可选的动态幅度（DyScope）
        self.dyscope = dyscope

    def _init_pos(self):
        """生成基准网格坐标（归一化到 [-1, 1]）。

        lp 模式下 offset 卷积输出 ``2*groups*scale*scale`` 个通道，
        因此基准坐标长度必须与之相同才能逐元素相加。
        做法：生成 scale×scale 个 (x, y) 基点，再为每个 group 复制一份。
        """
        h = torch.arange(-(self.scale - 1) / 2,
                         (self.scale - 1) / 2 + 1) / self.scale
        base = (torch.stack(torch.meshgrid([h, h], indexing="ij"))
                .flip(0).reshape(2, -1))            # (2, s*s)
        base = base.unsqueeze(1).repeat(1, self.groups, 1)  # (2, g, s*s)
        return base.reshape(-1, 1, 1).contiguous()  # (2*g*s*s, 1, 1)

    def _get_pos(self, x):
        """构造采样坐标 (b, H, W, 2)，H/W 为放大后的尺寸。

        输出分辨率**由 grid 的分辨率决定**——这是 ``grid_sample`` 的语义：
        grid 有多大，输出就有多大。因此必须显式构造分辨率为
        ``(h*scale, w*scale)`` 的网格，否则输出不会放大。

        两种模式的区别只在于偏移 ``o`` 的粒度：

        * ``pl``：每个源像素预测一组 (dx, dy)，其对应的 scale×scale 个
          输出点共用该偏移（用 repeat 展开）；
        * ``lp``：每个输出点独立预测一组偏移（先 reshape 成 s×s 再展开）。
        """
        h, w = x.shape[-2:]
        b = x.shape[0]
        s = self.scale
        H, W = h * s, w * s

        # 放大后的基准归一化网格 [-1, 1]
        dtype, device = x.dtype, x.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, dtype=dtype, device=device),
            torch.linspace(-1, 1, W, dtype=dtype, device=device),
            indexing="ij")
        pos = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

        if self.style == "pl":
            o = self.offset(x)                            # (b, 2g, h, w)
            o = o.permute(0, 2, 3, 1).reshape(
                b, h, w, 2, self.groups).mean(dim=-1)      # (b, h, w, 2)
            # 每个源像素的 s×s 个输出点共用该偏移。
            # (b, h, w, 2) -> (b, h, s, w, s, 2)：先插 s 到 h 后，再插 s 到 w 后
            o = o.unsqueeze(2).repeat(1, 1, s, 1, 1)        # (b, h, s, w, 2)
            o = o.unsqueeze(4).repeat(1, 1, 1, 1, s, 1)     # (b, h, s, w, s, 2)
            o = o.reshape(b, H, W, 2)
        else:  # "lp"
            o = self.offset(x) * 0.25 + self.init_pos      # (b, 2*g*s*s, h, w)
            if self.scope is not None:
                o = o * (torch.sigmoid(self.scope(x)) * 2.0 - 1.0)
            # (b, h, w, 2, g, s*s) -> 对 g 取平均 -> (b, h, w, 2, s*s)
            o = o.permute(0, 2, 3, 1).reshape(
                b, h, w, 2, self.groups, s * s).mean(dim=4)
            o = o.reshape(b, h, w, 2, s, s)                # (b, h, w, 2, s, s)
            # -> (b, h, s, w, s, 2)：把两个 s 维插到 h 与 w 之后
            o = o.permute(0, 1, 4, 3, 5, 2)
            o = o.reshape(b, H, W, 2)

        return pos + o * 2.0

    def forward(self, x):
        pos = self._get_pos(x)                             # (b, h, w, 2)
        out = F.grid_sample(
            x, pos.clamp(-1, 1), mode="bilinear",
            padding_mode="zeros", align_corners=False)
        return out


class ECA(nn.Module):
    """Efficient Channel Attention（无降维的通道注意力）。

    相比 SE / GCNet，ECA 用一维卷积直接建模通道间依赖，**不做降维**，
    因此参数量几乎为零（仅 kernel_size 个），却常有相近甚至更好的增益。
    适合加在 BiFPN 的融合节点上做轻量增强。
    """

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                 # (b, c, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (b, 1, c)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)
