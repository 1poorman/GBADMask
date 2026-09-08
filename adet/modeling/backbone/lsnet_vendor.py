# -*- coding: utf-8 -*-
"""LSNet（CVPR 2025, "See Large, Focus Small"）骨干 —— vendored 自
THU-MIG/lsnet 的 detection 移植版（commit 时点的纯 torch 语义），两处适配：

1. **SKA 去triton化**：官方 master 的 ska.py 已改为 triton kernel（本环境
   不可用）。此处按其数学语义重写为纯 torch：对 ks×ks 每个偏移，
   ``out += shift(x) * w[:, ci % wc, k]``，autograd 自动反传。数值等价。
2. **去 mmcv/mmdet 化**：注册器/logger/checkpoint 加载改由
   ``mobile_bb.py`` 的 d2 Backbone 包装层负责。
3. **Attention 变分辨率**：沿用 detection 版 forward 内对 attention bias
   的 bicubic 运行时插值（多尺度训练 480/512/544 下偏置表自动对齐）。

权重：lsnet_t.pth（ImageNet-1k，74.9 top-1，11.4M params）。"""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import SqueezeExcite


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                1, 1, 1, 1, device=x.device, dtype=x.dtype).ge_(self.drop).div(1 - self.drop)
        return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class Attention(torch.nn.Module):
    """LSNet stage-3 注意力。偏置表在任意输入分辨率下运行时 bicubic 插值。"""

    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
        attn = (q.transpose(-2, -1) @ k) * self.scale

        bias = self.attention_biases[:, self.attention_bias_idxs]
        bias = torch.nn.functional.interpolate(
            bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic')
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x


class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w


class SKA(torch.nn.Module):
    """Selective Kernel Aggregation（纯 torch 重实现，等价官方 triton 语义）。

    w: (n, wc, ks*ks, h, w)；x: (n, ic, h, w)。对每个偏移 (kh, kw)：
    out[..., hi, wi] += x[..., hi-pad+kh, wi-pad+kw] * w[:, ci % wc, kh*ks+kw, hi, wi]
    """

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n, ic, h, width = x.shape
        wc = w.shape[1]
        ks = int(math.sqrt(w.shape[2]))
        pad = (ks - 1) // 2
        xp = F.pad(x, (pad, pad, pad, pad))
        # 通道映射 ci % wc（wc 通常 = ic // 8）
        idx = torch.arange(ic, device=x.device) % wc
        w_sel = w[:, idx]  # (n, ic, ks*ks, h, w)
        out = None
        for kh in range(ks):
            for kw in range(ks):
                xs = xp[:, :, kh:kh + h, kw:kw + width]
                term = xs * w_sel[:, :, kh * ks + kw]
                out = term if out is None else out + term
        return out


class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x)) + x)


class Block(torch.nn.Module):
    def __init__(self, ed, kd, nh=8, ar=4, resolution=14, stage=-1, depth=-1):
        super().__init__()
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=14))
            else:
                self.mixer = LSConv(ed)

        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))


class LSNet(torch.nn.Module):
    """四阶段输出：blocks1..4 依次 stride 8/16/32/64（patch_embed 下采样 8×）。

    与分类版的差异：无 head（权重加载时跳过 head/head_dist/classifier），
    forward 直接返回 4 个 stage 特征元组。"""

    def __init__(self, img_size=512, patch_size=8, in_chans=3,
                 embed_dim=(64, 128, 256, 384), key_dim=(16, 16, 16, 16),
                 depth=(0, 2, 8, 10), num_heads=(3, 3, 3, 4)):
        super().__init__()
        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.patch_embed = torch.nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1),
        )
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]

        for i, (ed, kd, dpth, nh, ar) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            for d in range(dpth):
                blocks[i].append(Block(ed, kd, nh, ar, resolution, stage=i, depth=d))
            if i != len(depth) - 1:
                blk = blocks[i + 1]
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i], ks=3, stride=2, pad=1,
                                     groups=embed_dim[i]))
                blk.append(Conv2d_BN(embed_dim[i], embed_dim[i + 1], ks=1, stride=1, pad=0))
                resolution = (resolution - 1) // 2 + 1

        self.num_features = embed_dim[-1]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        x = self.blocks1(x)
        outs.append(x)
        x = self.blocks2(x)
        outs.append(x)
        x = self.blocks3(x)
        outs.append(x)
        x = self.blocks4(x)
        outs.append(x)
        return tuple(outs)


def lsnet_t(img_size=512, **kwargs):
    return LSNet(img_size=img_size, patch_size=8,
                 embed_dim=[64, 128, 256, 384], depth=[0, 2, 8, 10],
                 num_heads=[3, 3, 3, 4], **kwargs)


def lsnet_s(img_size=512, **kwargs):
    return LSNet(img_size=img_size, patch_size=8,
                 embed_dim=[96, 192, 320, 448], depth=[1, 2, 8, 10],
                 num_heads=[3, 3, 3, 4], **kwargs)


_LSNET_FACTORY = {"lsnet_t": lsnet_t, "lsnet_s": lsnet_s}
