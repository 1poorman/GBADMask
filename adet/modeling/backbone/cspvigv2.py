import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.layers import ShapeSpec

"""
MobileViGv2 + C3K2（C2f 风格）融合骨干
=====================================

思路
----
1. 直接复用 MobileViGv2（./models/MobileViGv2）的核心算子：``Stem``、
   ``DepthWiseSeparable``、``InvertedResidual``（用 DepthWiseSeparable 替代 MLP）、
   5 连接（5-connection）``MRConv4d``、``RepCPE``、``Grapher``、``MGC``、``Downsample``。
   这些子模块的命名严格保持与官方预训练权重一致，从而可以 1:1 映射。

2. 参考 cspnet + mobilevig 的融合方式，但改为 **C3K2（即 C2f）融合**：
   每个 stage 的若干 block 在 **整通道** 上串行（与预训练权重通道数一致，
   必须整通道才能直接映射），把所有 block 的输出 + 原始输入（skip）沿通道
   concat 后，用 1×1 融合。这与 YOLOv8/v11 的 C2f / C3K2 结构一致——
   每个分支的输出都保留，梯度可直接流向每个 block，特征复用更强。

   为什么不用原 cspvig 的“通道对半分”方式？
   原方式把 block 跑在 ``ch//2`` 上，而官方 MobileViGv2 权重是 ``ch`` 整通道，
   无法 1:1 映射。为了“新骨干的权重映射”成立，这里 block 必须整通道，
   fusion 仅作为新增的 1×1 分支（随机/恒等初始化，不影响预训练权重加载）。

3. 融合层初始化为 **恒等直通最后一个 block 的输出**，因此加载预训练权重后，
   前向输出与原始 MobileViGv2 完全等价 —— 预训练特征在初始化时刻即“有效”，
   训练时融合层再逐步利用多分支信息微调。

权重映射
--------
官方 checkpoint 顶层为 ``state_dict`` / ``state_dict_ema``，内部键为
``stem.stem.*`` 与 ``backbone.{i}.{j}.*``，以及分类头 ``prediction/head/dist_head``。
映射规则：
- ``stem.*``         -> ``stem.*``（完全一致）
- ``backbone.{i}.*`` -> ``stages.{i}.*``（仅改前缀）
- ``prediction/*/dist_head`` -> 丢弃（检测任务不需要）
- 融合层 ``fuse.{i}.*`` -> 不在预训练里，保留融合层自身初始化
"""

# --------------------------------------------------------------------------- #
# MobileViGv2 核心算子（命名与官方预训练权重一致，便于 1:1 映射）
# --------------------------------------------------------------------------- #
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class DepthWiseSeparable(nn.Module):
    """MobileViGv2 的 InvertedResidual 中间块：逐点升维 -> 深度可分离 3x3 -> 逐点降维。"""

    def __init__(self, in_dim, kernel, e=4):
        super().__init__()
        e = int(e)
        self.pw1 = nn.Conv2d(in_dim, in_dim * e, 1)
        self.norm1 = nn.BatchNorm2d(in_dim * e)
        self.act1 = nn.GELU()

        self.dw = nn.Conv2d(in_dim * e, in_dim * e, kernel_size=kernel,
                            stride=1, padding=1, groups=in_dim * e)
        self.norm2 = nn.BatchNorm2d(in_dim * e)
        self.act2 = nn.GELU()

        self.pw2 = nn.Conv2d(in_dim * e, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.act1(self.norm1(self.pw1(x)))
        x = self.act2(self.norm2(self.dw(x)))
        x = self.norm3(self.pw2(x))
        return x


class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, e=expansion_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            return x + self.drop_path(self.layer_scale_1 * self.dws(x))
        return x + self.drop_path(self.dws(x))


class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution（5-connection 图构建，来自 MobileViGv2）。
    K 为跳数（hop distance），通过对特征图做 ±K 的循环平移构造 4 个相对位置，
    与自身做 max-relative 得到 ``x_j``，再与 ``x`` 拼接后经 1×1 卷积。
    """

    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape

        x_j = x - x
        # 上 / 下两个方向
        x_c = torch.cat([x[:, :, -self.K:, :], x[:, :, :-self.K, :]], dim=2)
        x_j = torch.max(x_j, x_c - x)
        x_c = torch.cat([x[:, :, self.K:, :], x[:, :, :self.K, :]], dim=2)
        x_j = torch.max(x_j, x_c - x)
        # 左 / 右两个方向
        x_r = torch.cat([x[:, :, :, -self.K:], x[:, :, :, :-self.K]], dim=3)
        x_j = torch.max(x_j, x_r - x)
        x_r = torch.cat([x[:, :, :, self.K:], x[:, :, :, :self.K]], dim=3)
        x_j = torch.max(x_j, x_r - x)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class RepCPE(nn.Module):
    """可重参数化的条件位置编码（训练期用带 bias 的深度卷积，推理期可融合）。"""

    def __init__(self, in_channels, embed_dim, spatial_shape=(7, 7), inference_mode=False):
        super(RepCPE, self).__init__()
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.embed_dim,
                kernel_size=self.spatial_shape, stride=1,
                padding=int(self.spatial_shape[0] // 2), groups=self.embed_dim, bias=True)
        else:
            self.pe = nn.Conv2d(
                in_channels, embed_dim, spatial_shape, 1,
                int(spatial_shape[0] // 2), bias=True, groups=embed_dim)

    def forward(self, x):
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)
        return self.pe(x) + x


class Grapher(nn.Module):
    """图卷积混合器：CPE -> fc1 -> MRConv4d(5-connection) -> fc2。"""

    def __init__(self, in_channels, K):
        super(Grapher, self).__init__()
        self.cpe = RepCPE(in_channels=in_channels, embed_dim=in_channels, spatial_shape=(7, 7))
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels * 2, in_channels, K=K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return x


class MGC(nn.Module):
    """Mixer(Grapher) + FFN，各带 layer_scale 残差。"""

    def __init__(self, in_dim, drop_path=0., K=2, use_layer_scale=True,
                 layer_scale_init_value=1e-5):
        super().__init__()
        self.mixer = Grapher(in_dim, K)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((in_dim, 1, 1)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((in_dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------------------------------- #
# C3K2（C2f 风格）融合层
# --------------------------------------------------------------------------- #
class C3K2Fusion(nn.Module):
    """
    C3K2 / C2f 风格的融合：把 ``[skip, b1, b2, ..., b_n]`` 沿通道 concat 后用 1×1 卷积融合。

    - ``n`` 为该 stage 的 block 数量；输入通道 = ``(n + 1) * ch``，输出通道 = ``ch``。
    - 初始化为 **恒等直通最后一个 block 输出**（即 ``b_n``），故加载预训练权重后
      前向与原始 MobileViGv2 等价；训练时融合层再利用其余分支自适应调整。
    """

    def __init__(self, ch, n, out_ch=None):
        super().__init__()
        out_ch = out_ch or ch
        self.n = n
        self.conv = nn.Conv2d(ch * (n + 1), out_ch, 1, bias=False)
        self._init_passthrough(ch, out_ch)

    def _init_passthrough(self, ch, out_ch):
        # 令输出 = concat 中最后一个张量（b_n，索引 = n）的恒等映射
        nn.init.zeros_(self.conv.weight)
        for c in range(out_ch):
            self.conv.weight.data[c, self.n * ch + c, 0, 0] = 1.0

    def forward(self, feats):
        return self.conv(torch.cat(feats, dim=1))


# --------------------------------------------------------------------------- #
# 变体定义（与官方 mobilevigv2_{ti,s,m,b} 一致）
# --------------------------------------------------------------------------- #
VARIANTS = {
    "ti": dict(blocks=[[2, 0], [2, 2], [6, 2], [2, 2]], channels=[32, 64, 128, 256]),
    "s":  dict(blocks=[[3, 0], [3, 3], [9, 3], [3, 3]], channels=[32, 64, 128, 256]),
    "m":  dict(blocks=[[3, 0], [3, 3], [9, 3], [3, 3]], channels=[32, 64, 192, 384]),
    "b":  dict(blocks=[[3, 0], [3, 3], [9, 3], [3, 3]], channels=[64, 128, 256, 512]),
}


class MobileViGv2_CSP(Backbone):
    def __init__(self, blocks, channels, drop_path=0.1, K=(0, 8, 4, 2),
                 use_c3k2=True, out_features=("res2", "res3", "res4", "res5"),
                 pretrained=None):
        super(MobileViGv2_CSP, self).__init__()

        self.use_c3k2 = use_c3k2
        self._out_features = list(out_features)

        # 与官方一致：每个 block（IR 与 MGC 各占一个 drop rate）分配线性衰减的 drop_path
        n_blocks = sum(sum(x) for x in blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]
        dpr_idx = 0

        self.stem = Stem(input_dim=3, output_dim=channels[0])

        # 每个 stage 构造成 nn.Sequential，子模块命名与官方 backbone.{i}.{j} 完全一致，
        # 这样权重映射只需把前缀 backbone -> stages。
        self.stages = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(len(blocks)):
            local_stages = blocks[i][0]
            global_stages = blocks[i][1]
            stage = []
            if i > 0:
                stage.append(Downsample(channels[i - 1], channels[i]))
            for _ in range(local_stages):
                stage.append(InvertedResidual(dim=channels[i], kernel=3,
                                              drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages):
                stage.append(MGC(channels[i], drop_path=dpr[dpr_idx], K=K[i]))
                dpr_idx += 1
            self.stages.append(nn.Sequential(*stage))
            self.fuse.append(C3K2Fusion(channels[i], local_stages + global_stages))

        self._out_feature_channels = {
            "res{}".format(j + 2): channels[j] for j in range(len(channels))
        }
        self._out_feature_strides = {
            "res{}".format(j + 2): 2 ** (j + 2) for j in range(len(channels))
        }

        if pretrained:
            self.load_mobilevigv2_pretrained(pretrained)
        else:
            self._initialize_weights()

    # ------------------------------------------------------------------ #
    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        for i in range(len(self.stages)):
            x_in = x
            feats = []
            y = x
            for m in self.stages[i]:
                y = m(y)
                feats.append(y)
            if self.use_c3k2:
                # skip：i>0 时为下采样输出（已对齐到 ch_i），i==0 时为 stem 输出
                if i > 0:
                    skip = feats[0]
                    block_feats = feats[1:]
                else:
                    skip = x_in
                    block_feats = feats
                x = self.fuse[i]([skip] + block_feats)
            else:
                x = y  # 不使用融合，等价于原始 MobileViGv2 串行堆叠
            outs.append(x)
        return {"res{}".format(j + 2): r for j, r in enumerate(outs)}

    # ------------------------------------------------------------------ #
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ------------------------------------------------------------------ #
    def load_mobilevigv2_pretrained(self, path, use_ema=True):
        """
        从官方 MobileViGv2 分类预训练权重加载骨干部分。
        返回 {loaded, skipped, mismatched} 统计，便于核对映射是否完整。
        """
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict_ema" in ckpt and use_ema:
                sd = ckpt["state_dict_ema"]
            elif "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            elif "state_dict_ema" in ckpt:
                sd = ckpt["state_dict_ema"]
            else:
                sd = ckpt
        else:
            sd = ckpt
        # 只保留张量，剔除可能存在的非参数条目
        sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

        model_sd = self.state_dict()
        loaded, skipped, mismatched = 0, [], []
        for k, v in sd.items():
            # 分类头不在检测骨干里，跳过
            if k.startswith("prediction") or k.startswith("head") or k.startswith("dist_head"):
                skipped.append(k)
                continue
            # 前缀映射：backbone.{i}.{j} -> stages.{i}.{j}；stem.* 保持不变
            if k.startswith("backbone."):
                nk = "stages." + k[len("backbone."):]
            elif k.startswith("stem."):
                nk = k
            else:
                skipped.append(k)
                continue

            if nk in model_sd:
                if model_sd[nk].shape == v.shape:
                    model_sd[nk] = v
                    loaded += 1
                else:
                    mismatched.append((k, nk, tuple(v.shape), tuple(model_sd[nk].shape)))
            else:
                skipped.append(k)

        self.load_state_dict(model_sd, strict=False)
        # 融合层不在预训练权重中，保持 _init_passthrough 的恒等初始化
        return {"loaded": loaded, "skipped": skipped, "mismatched": mismatched}


# --------------------------------------------------------------------------- #
# 注册：仅骨干
# --------------------------------------------------------------------------- #
@BACKBONE_REGISTRY.register()
def build_mobilevigv2_csp_backbone(cfg, input_shape):
    version = cfg.MODEL.VIG.VERSION.lower()
    if version not in VARIANTS:
        raise ValueError("MODEL.VIG.VERSION 必须是 {} 之一，收到 {}".format(
            list(VARIANTS.keys()), version))
    variant = VARIANTS[version]
    model = MobileViGv2_CSP(
        blocks=variant["blocks"],
        channels=variant["channels"],
        drop_path=cfg.MODEL.VIG.DROP_PATH,
        use_c3k2=cfg.MODEL.VIG.USE_C3K2,
        out_features=cfg.MODEL.RESNETS.OUT_FEATURES,
        pretrained=cfg.MODEL.VIG.PRETRAINED or None,
    )
    return model


# --------------------------------------------------------------------------- #
# 注册：MobileViGv2_CSP + BiFPN（与 build_fcos_cspvig_bifpn_backbone 对应）
# --------------------------------------------------------------------------- #
@BACKBONE_REGISTRY.register()
def build_fcos_mobilevigv2_csp_bifpn_backbone(cfg, input_shape: ShapeSpec):
    from .bifpn import BiFPN

    bottom_up = build_mobilevigv2_csp_backbone(cfg, input_shape)
    in_features = cfg.MODEL.BiFPN.IN_FEATURES
    out_channels = cfg.MODEL.BiFPN.OUT_CHANNELS
    num_repeats = cfg.MODEL.BiFPN.NUM_REPEATS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS

    backbone = BiFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_top_levels=top_levels,
        num_repeats=num_repeats,
        norm=cfg.MODEL.BiFPN.NORM,
        upsample=cfg.MODEL.BiFPN.UPSAMPLE,
        attn=cfg.MODEL.BiFPN.ATTN,
    )
    return backbone
