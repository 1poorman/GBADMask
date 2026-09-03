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
- 内层 ``Attention`` 采用**标准缩放点积多头自注意力**（ViT/DETR 同款）。
  相比 RT-DETR 原版线性注意力，标准 MHA 数值更稳定、实现更易验证，且对
  basis tower 这类中小分辨率特征图（~80×80）计算量完全可控。语义上它正是
  "对空间 token 做位置敏感加权"这一 C2PSA 核心思想。
- 该模块**不引入额外下采样/上采样**，可直接挂到 ``basis_module`` 的
  ``attn_low`` / ``attn_tower`` 位置（``MODEL.BASIS_MODULE.ATTN = "psa"``）。

注意：PSA 作用在 2D 特征图上，内部会把空间展平成 token 做注意力，因此
``num_heads`` 必须整除 ``channels``；这里自动选择可整除的最大头数。
"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from adet.layers import conv_with_kaiming_uniform


def _pick_heads(channels: int, preferred: int = 8) -> int:
    for h in (preferred, 4, 2, 1):
        if channels % h == 0:
            return h
    return 1


class Attention(nn.Module):
    """标准缩放点积多头自注意力，作用于 (B, C, H, W) 的空间 token。"""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "num_heads 必须整除 dim"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        # (B, C, H, W) -> (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, nh, N, hd)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)                              # (B, N, C)
        out = out.transpose(1, 2).reshape(B, C, H, W)     # (B, C, H, W)
        return out


class PSA(nn.Module):
    """Position-Sensitive Attention 块（C2PSA 风格）。

    Args:
        c1, c2: 输入/输出通道数，必须相等。
        e: 分支宽度系数，隐通道 ``c_ = int(c1 * e)``（默认 0.5，与 RT-DETR 一致）。
        num_heads: 注意力头数（默认 8，自动向下取可整除值）。
        norm: 内部卷积的归一化类型，与 basis 模块保持一致（默认 "SyncBN"）。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        e: float = 0.5,
        num_heads: int = 8,
        norm: Optional[str] = "SyncBN",
    ):
        super().__init__()
        assert c1 == c2, "PSA 要求 c1 == c2"
        c_ = int(c1 * e)
        self.c_ = c_
        conv = conv_with_kaiming_uniform(norm, True)
        self.conv1 = conv(c1, 2 * c_, 1, 1)   # 1×1：把通道拆成两半
        self.conv2 = conv(2 * c_, c1, 1, 1)   # 1×1：两半拼回原通道
        self.attn = Attention(c_, _pick_heads(c_, num_heads))
        self.ffn = nn.Sequential(
            conv(c_, c_, 1, 1),
            conv(c_, c_, 3, 1),
            conv(c_, c_, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.conv1(x).split((self.c_, self.c_), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.conv2(torch.cat((a, b), dim=1))
