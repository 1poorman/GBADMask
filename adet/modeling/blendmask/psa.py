# Copyright (c) GBADMask. All Rights Reserved.
"""
PSA (Position-Sensitive Attention) 模块 —— 对应 ROADMAP M6.2 的 B1 组件。

设计来源
--------
- RT-DETR 的 ``PSA`` 块（ECCV 2024），以及 Ultralytics YOLO11 / YOLO26 中
  沿用的 ``C2PSA`` 结构。核心思想：把通道分成两半，仅对其中一半依次施加
  **位置敏感注意力** 与 **前馈网络**，再与另一半拼接，从而在几乎不增加
  参数量的前提下注入全局/位置感知能力。

实现说明
--------
- 外层 wrapper（``PSA``）严格复刻 RT-DETR 的 split-conv-attn-ffn-concat 结构，
  要求 ``c1 == c2``，与项目 ``build_attention(name, channels)`` 的接口一致
  （输入/输出均为 ``(B, C, H, W)`` 且通道不变）。
- 内层 ``Attention`` 采用**窗口式（window）缩放点积多头自注意力**：
  特征图切成 ``window``×``window`` 的不重叠窗口，注意力只在窗口内计算。
  这正是 RT-DETR / C2PSA 的原生设计（其对 1/8 特征用 window=14, 25 个窗口
  拼接后做注意力），相比全局注意力：
  * FLOPs/显存从 O(N²) 降到 O(N·w²)（w=窗口边长），160×160 低层分支
    从 ~84GB 全局矩阵降为 196 个 14×14 窗口的局部矩阵（~MB 级）；
  * 短距离依赖（病斑边界、纹理）本来就是 bases 最需要的信号；
  * 不依赖 SDPA 的 fused kernel —— torch 2.0.1 训练模式下 SDPA 存在
    kernel 回退问题（实测 4.9 s/iter，8.4×减速），窗口实现直接用
    matmul 即可保持全速。
- 特征图边长非窗口整数倍时，边缘零填充到整数倍，注意力输出只取有效区域
  （填充 token 只互相 attend，不影响有效 token 的输出）。
- 该模块**不引入额外下采样/上采样**，可直接挂到 ``basis_module`` 的
  ``attn_low`` / ``attn_tower`` 位置（``MODEL.BASIS_MODULE.ATTN = "psa"``）。
"""
import torch
from typing import Optional
from torch import nn
from torch.nn import functional as F

from adet.layers import conv_with_kaiming_uniform


def _pick_heads(dim: int, preferred: int = 8) -> int:
    """在 preferred 以内选一个能整除 dim 的最大头数。"""
    for h in (preferred, 4, 2, 1):
        if dim % h == 0:
            return h
    return 1


class Attention(nn.Module):
    """窗口式缩放点积多头自注意力，作用于 (B, C, H, W) 的空间 token。

    Args:
        dim: 通道数（注意力输入通道）。
        num_heads: 头数，需整除 dim。
        window: 窗口边长（默认 14，与 RT-DETR PSA 在 1/8 特征上的选择一致）。
    """

    def __init__(self, dim: int, num_heads: int = 8, window: int = 14):
        super().__init__()
        assert dim % num_heads == 0, "num_heads 必须整除 dim"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.window
        # 边缘零填充到窗口整数倍（pad 行/列只在 pad 内互相 attend，
        # 不影响有效 token：pad 的 q/k 为 0 → 得分 0 → 与有效 token 得分
        # 混在一起但有效 token 间的相对得分不变；严格隔离见下方掩码）
        pad_h = (-H) % w
        pad_w = (-W) % w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // w, Wp // w

        # (B, C, Hp, Wp) -> (B, nh, nH*nW, w*w, hd)
        x = x.reshape(B, C, nH, w, nW, w).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B * nH * nW, w * w, C)
        qkv = self.qkv(x).reshape(B * nH * nW, w * w, 3,
                                  self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, Bw, nh, w², hd)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (Bw, nh, w², hd)

        # pad token 屏蔽：让 pad 行均匀 attend 自己（不影响有效行），
        # 有效行不 attend pad 列
        if pad_h or pad_w:
            valid = torch.zeros(Hp, Wp, dtype=torch.bool, device=x.device)
            valid[:H, :W] = True
            # (nH*w, nW*w) -> (nH, w, nW, w) -> (nW*nH, w*w) 每窗口有效位
            # （batch 维共享同一掩码，广播到 (B*nH*nW, 1, w², w²)）
            valid = valid.reshape(nH, w, nW, w).permute(0, 2, 1, 3)
            valid = valid.reshape(nH * nW, w * w)
            mask = (valid[:, :, None] | valid[:, None, :]).unsqueeze(1)
            mask = mask.unsqueeze(0).expand(B, -1, -1, -1, -1)
            mask = mask.reshape(B * nH * nW, 1, w * w, w * w)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(~mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            # 全 pad 行（mask 全 False）softmax 会得 NaN，置零
            attn = torch.nan_to_num(attn, nan=0.0)
            out = attn @ v
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = out.permute(0, 2, 1, 3).reshape(B * nH * nW, w * w, C)
        out = self.proj(out)
        out = out.reshape(B, nH, nW, w, w, C).permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(B, C, Hp, Wp)
        return out[..., :H, :W]


class PSA(nn.Module):
    """Position-Sensitive Attention 块（C2PSA 风格）。

    Args:
        c1, c2: 输入/输出通道数，必须相等。
        e: 分支宽度系数，隐通道 ``c_ = int(c1 * e)``（默认 0.5，与 RT-DETR 一致）。
        num_heads: 注意力头数（默认 8，自动向下取可整除值）。
        window: 注意力窗口边长（默认 14）。
        norm: 内部卷积的归一化类型，与 basis 模块保持一致（默认 "SyncBN"）。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        e: float = 0.5,
        num_heads: int = 8,
        window: int = 14,
        norm: Optional[str] = "SyncBN",
        use_checkpoint: bool = True,
    ):
        super().__init__()
        assert c1 == c2, "PSA 要求 c1 == c2"
        c_ = int(c1 * e)
        self.c_ = c_
        conv = conv_with_kaiming_uniform(norm, True)
        self.conv1 = conv(c1, 2 * c_, 1, 1)   # 1×1：把通道拆成两半
        self.conv2 = conv(2 * c_, c1, 1, 1)   # 1×1：两半拼回原通道
        self.attn = Attention(c_, _pick_heads(c_, num_heads), window=window)
        self.ffn = nn.Sequential(
            conv(c_, c_, 1, 1),
            conv(c_, c_, 3, 1),
            conv(c_, c_, 1, 1),
        )
        # 注意力反向需暂存 (B·窗口数, nh, w², w²) 的 attn 矩阵（batch 7 下
        # 两插入点合计 ~2GB×2 份），在小显存预算（GPU1 可用 ~16GB）下会 OOM。
        # 梯度检查点用一次前向重算换掉这些暂存，显存换算力（约 +20% 时间）。
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.conv1(x).split((self.c_, self.c_), dim=1)
        if self.use_checkpoint and self.training and x.requires_grad:
            b = b + torch.utils.checkpoint.checkpoint(
                self.attn, b, use_reentrant=False)
        else:
            b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.conv2(torch.cat((a, b), dim=1))
