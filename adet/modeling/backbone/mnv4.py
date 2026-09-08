# -*- coding: utf-8 -*-
"""MobileNetV4-Conv-Small（arXiv:2404.10518）—— 按 timm 官方 checkpoint
键名精确复刻的自包含实现（不依赖新版 timm 的 builder/blocks）。

规格来源：timm master ``mobilenetv3.py::_gen_mobilenet_v4('mobilenetv4_conv_small')``
的 arch_def（timm 的 'uir' 解码规则：a=起始 dw 核、k=中间 dw 核、p=末端 dw 核、
e=扩张比、c=输出通道、s=stride；MNv4-S 全部块无 'n' 激活标记 → 模型默认 ReLU）。
参数名与 ``mobilenetv4_conv_small.e2400_r224_in1k`` checkpoint 逐键一致：
stem 为裸 conv_stem/bn1，cn 块为 blocks.N.M.{conv,bn1}，UIB 内部为
{dw_start,pw_exp,dw_mid,pw_proj}.{conv,bn}，尾部 conv_head/norm_head/classifier。

输出 stage：blocks.1 (64ch, stride 8) / blocks.2 (96ch, stride 16) /
blocks.3 (128ch, stride 32) —— 供 BiFPN 的 res3/res4/res5 抽头。"""
import torch
import torch.nn as nn


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 保证下降幅度不超过 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBnAct(nn.Module):
    """timm ConvNormAct：conv + bn（可选 act）。bn_name 决定子模块名
    （cn 块用 bn1，UIB 内部用 bn，与 checkpoint 键名一致）。"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 groups=1, apply_act=True, bn_name="bn"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              groups=groups, bias=False)
        setattr(self, bn_name, nn.BatchNorm2d(out_ch))
        self._bn_name = bn_name
        self.apply_act = apply_act

    def forward(self, x):
        x = self.conv(x)
        x = getattr(self, self._bn_name)(x)
        if self.apply_act:
            x = torch.nn.functional.relu(x)
        return x


class UIB(nn.Module):
    """Universal Inverted Residual（ExtraDW/IR/ConvNeXt 三形态由核参数决定）。

    前向顺序（timm）：dw_start -> pw_exp -> dw_mid -> (se) -> pw_proj (+残差)。
    stride 由 dw_mid 承接（dw_start 存在时 stride=1，与 timm 语义一致）。"""

    def __init__(self, in_chs, out_chs, dw_ks_start=0, dw_ks_mid=3,
                 stride=1, exp_ratio=1.0):
        super().__init__()
        mid_chs = _make_divisible(in_chs * exp_ratio)
        self.has_skip = (in_chs == out_chs and stride == 1)

        if dw_ks_start:
            self.dw_start = ConvBnAct(in_chs, in_chs, dw_ks_start, stride=1,
                                      padding=dw_ks_start // 2, groups=in_chs,
                                      apply_act=False)
        else:
            self.dw_start = nn.Identity()

        self.pw_exp = ConvBnAct(in_chs, mid_chs, 1)

        if dw_ks_mid:
            self.dw_mid = ConvBnAct(mid_chs, mid_chs, dw_ks_mid, stride=stride,
                                    padding=dw_ks_mid // 2, groups=mid_chs)
        else:
            self.dw_mid = nn.Identity()

        self.pw_proj = ConvBnAct(mid_chs, out_chs, 1, apply_act=False)

    def forward(self, x):
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.pw_proj(x)
        if self.has_skip:
            x = x + shortcut
        return x


class _CnBlock(ConvBnAct):
    """'cn' 普通卷积块（checkpoint 命名 conv/bn1）。"""

    def __init__(self, in_chs, out_chs, kernel_size, stride=1):
        super().__init__(in_chs, out_chs, kernel_size, stride=stride,
                         padding=kernel_size // 2, bn_name="bn1")


class MobileNetV4Small(nn.Module):
    """mobilenetv4_conv_small（1.0×）。四输出：stride 8/16/32/32。"""

    def __init__(self, num_classes=1000, in_chans=3):
        super().__init__()
        # stem：3->32, k3 s2
        self.conv_stem = nn.Conv2d(in_chans, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        def s2():
            # in 64
            return [
                # 'uir_r1_a5_k5_s2_e3_c96'  ExtraDW, dw_start k5, dw_mid k5 s2
                UIB(64, 96, dw_ks_start=5, dw_ks_mid=5, stride=2, exp_ratio=3.0),
                # 'uir_r4_a0_k3_s1_e2_c96'  IR ×4
                UIB(96, 96, dw_ks_mid=3, stride=1, exp_ratio=2.0),
                UIB(96, 96, dw_ks_mid=3, stride=1, exp_ratio=2.0),
                UIB(96, 96, dw_ks_mid=3, stride=1, exp_ratio=2.0),
                UIB(96, 96, dw_ks_mid=3, stride=1, exp_ratio=2.0),
                # 'uir_r1_a3_k0_s1_e4_c96'  ConvNeXt：dw_start k3，无 dw_mid
                UIB(96, 96, dw_ks_start=3, dw_ks_mid=0, stride=1, exp_ratio=4.0),
            ]

        def s3():
            # in 96
            return [
                # 'uir_r1_a3_k3_s2_e6_c128'
                UIB(96, 128, dw_ks_start=3, dw_ks_mid=3, stride=2, exp_ratio=6.0),
                # 'uir_r1_a5_k5_s1_e4_c128'
                UIB(128, 128, dw_ks_start=5, dw_ks_mid=5, stride=1, exp_ratio=4.0),
                # 'uir_r1_a0_k5_s1_e4_c128'
                UIB(128, 128, dw_ks_mid=5, stride=1, exp_ratio=4.0),
                # 'uir_r1_a0_k5_s1_e3_c128'
                UIB(128, 128, dw_ks_mid=5, stride=1, exp_ratio=3.0),
                # 'uir_r2_a0_k3_s1_e4_c128' ×2
                UIB(128, 128, dw_ks_mid=3, stride=1, exp_ratio=4.0),
                UIB(128, 128, dw_ks_mid=3, stride=1, exp_ratio=4.0),
            ]

        self.blocks = nn.ModuleList([
            # stage0：conv 32->32 k3 s2、conv 32->32 k1（'cn' 块）
            nn.Sequential(
                _CnBlock(32, 32, 3, stride=2),
                _CnBlock(32, 32, 1),
            ),
            # stage1：conv 32->96 k3 s2、conv 96->64 k1
            nn.Sequential(
                _CnBlock(32, 96, 3, stride=2),
                _CnBlock(96, 64, 1),
            ),
            # stage2（stride 16，出 96ch）
            nn.Sequential(*s2()),
            # stage3（stride 32，出 128ch）
            nn.Sequential(*s3()),
            # stage4：conv 128->960 k1
            nn.Sequential(_CnBlock(128, 960, 1)),
        ])

        self.conv_head = nn.Conv2d(960, 1280, 1, bias=False)
        self.norm_head = nn.BatchNorm2d(1280)
        self.classifier = nn.Linear(1280, num_classes)

    def forward_features(self, x):
        """返回 (blocks.1, blocks.2, blocks.3) 输出 = stride 8/16/32。"""
        x = torch.nn.functional.relu(self.bn1(self.conv_stem(x)))
        for i, stage in enumerate(self.blocks[:4]):
            x = stage(x)
            if i == 1:
                f8 = x
            elif i == 2:
                f16 = x
        f32 = x
        return f8, f16, f32

    def forward(self, x):
        f8, f16, f32 = self.forward_features(x)
        x = self.blocks[4](f32)
        x = torch.nn.functional.relu(self.norm_head(self.conv_head(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


_MNV4_FACTORY = {"mobilenetv4_conv_small": MobileNetV4Small}
