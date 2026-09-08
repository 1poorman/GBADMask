# -*- coding: utf-8 -*-
"""通用轻量骨干接入层（M6.6 骨干消融）：MobileNetV3-L / MobileNetV4-Conv-S /
LSNet-T 统一包装成 detectron2 Backbone，输出 res3/res4/res5（stride 8/16/32）
供 BiFPN(3,160) 消费——与 cspvigv2 平台完全同颈同协议。

- MobileNetV3-L：本地 timm 0.6.12 原生 ``mobilenetv3_large_100``
  （features_only），权重 mobilenetv3_large_100_ra IN-1k。
- MobileNetV4-Conv-S：``adet/modeling/backbone/mnv4.py``（键名精确复刻 timm
  checkpoint），权重 mobilenetv4_conv_small.e2400 IN-1k。
- LSNet-T：``adet/modeling/backbone/lsnet_vendor.py``（CVPR2025 检测移植版，
  SKA 纯 torch 重写 + attention bias 运行时插值），权重 lsnet_t IN-1k。

预训练加载为「宽容加载 + 覆盖率断言」：允许 head/classifier 等多余键，但
骨干体（blocks/stem/embed）必须 ≥95% 命中，防映射错误静默漏载。"""
import logging
import math

import torch

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec

from .lsnet_vendor import _LSNET_FACTORY
from .mnv4 import _MNV4_FACTORY

logger = logging.getLogger(__name__)


class MobileBBBackbone(Backbone):
    """轻量骨干统一包装。

    Args:
        model_name: "mnv3_l" / "mnv4_s" / "lsnet_t"
        weights: ImageNet checkpoint 路径（空 = 随机初始化，仅冒烟用）
        out_features: 输出层名（默认 res3/res4/res5）
    """

    SUPPORTED = ("mnv3_l", "mnv4_s", "lsnet_t")

    def __init__(self, model_name, weights="", input_size=512,
                 out_features=("res3", "res4", "res5")):
        super().__init__()
        assert model_name in self.SUPPORTED, \
            "MODEL.MOBILE_BB.MODEL_NAME 必须是 {} 之一，收到 {}".format(
                self.SUPPORTED, model_name)
        self.model_name = model_name

        if model_name == "mnv3_l":
            import timm
            self.net = timm.create_model(
                "mobilenetv3_large_100", features_only=True,
                out_indices=(0, 1, 2, 3, 4), pretrained=False)
            # timm 0.6.12 feature_info：reduction 2/4/8/16/32 共 5 tap
            info = self.net.feature_info
            stages = []
            for fi in info:
                stages.append((fi["num_chs"], fi["reduction"]))
            # 按需求挑 stride 8/16/32 的三个 tap
            picks = {}
            for idx, (ch, red) in enumerate(stages):
                if red in (8, 16, 32) and red not in picks:
                    picks[red] = idx
            assert sorted(picks) == [8, 16, 32], \
                "mobilenetv3 feature_info 缺 stride 8/16/32：{}".format(stages)
            self._picks = [picks[8], picks[16], picks[32]]
            self._channels = {
                "res3": stages[self._picks[0]][0],
                "res4": stages[self._picks[1]][0],
                "res5": stages[self._picks[2]][0],
            }
            self._strides = {"res3": 8, "res4": 16, "res5": 32}
            self._forward_impl = self._forward_timm
        elif model_name == "mnv4_s":
            self.net = _MNV4_FACTORY["mobilenetv4_conv_small"]()
            self._channels = {"res3": 64, "res4": 96, "res5": 128}
            self._strides = {"res3": 8, "res4": 16, "res5": 32}
            self._forward_impl = self._forward_mnv4
        else:  # lsnet_t / lsnet_s
            self.net = _LSNET_FACTORY[model_name](img_size=input_size)
            outs = self.net(torch.zeros(1, 3, input_size, input_size))
            chs = [o.shape[1] for o in outs]
            # outs = stride 8/16/32/64
            self._channels = {"res3": chs[0], "res4": chs[1], "res5": chs[2]}
            self._strides = {"res3": 8, "res4": 16, "res5": 32}
            self._forward_impl = self._forward_lsnet

        self._out_features = list(out_features)
        self._out_feature_channels = {k: self._channels[k] for k in self._out_features}
        self._out_feature_strides = {k: self._strides[k] for k in self._out_features}
        self._size_divisibility = 32

        if weights:
            stats = self._load_pretrained(weights)
            logger.info("[MobileBB:{}] 预训练加载 {}".format(model_name, stats))

    # ------------------------------------------------------------------ #
    # 权重加载
    # ------------------------------------------------------------------ #
    def _load_pretrained(self, weights):
        sd = torch.load(weights, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if self.model_name.startswith("lsnet"):
            # LSNet：删除索引 buffer，注意力偏置表按当前分辨率双三次插值
            for k in [k for k in sd if "attention_bias_idxs" in k]:
                del sd[k]
            msd = self.net.state_dict()
            for k in [k for k in sd if "attention_biases" in k]:
                pre = sd[k]
                cur = msd[k]
                nH1, L1 = pre.shape
                nH2, L2 = cur.shape
                if nH1 == nH2 and L1 != L2:
                    S1, S2 = int(L1 ** 0.5), int(L2 ** 0.5)
                    pre = torch.nn.functional.interpolate(
                        pre.view(1, nH1, S1, S1), size=(S2, S2),
                        mode="bicubic").view(nH2, L2)
                    sd[k] = pre
        model_sd = self.net.state_dict()
        loaded, skipped = 0, []
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded += 1
            else:
                skipped.append(k)
        # 覆盖率断言：骨干体必须基本命中（只允许 head/classifier 类多余键）
        n_have = len([k for k in model_sd])
        cover = loaded / max(n_have, 1)
        assert cover >= 0.95, \
            "[MobileBB:{}] 预训练覆盖率仅 {:.2f}（{} 加载 / {} 模型键），" \
            "skipped 样例 {} —— 权重映射可能错位".format(
                self.model_name, cover, loaded, n_have, skipped[:8])
        self.net.load_state_dict(model_sd)
        return {"loaded": loaded, "model_keys": n_have, "skipped": skipped[:8],
                "coverage": round(cover, 3)}

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def _forward_timm(self, x):
        feats = self.net(x)
        return feats[self._picks[0]], feats[self._picks[1]], feats[self._picks[2]]

    def _forward_mnv4(self, x):
        return self.net.forward_features(x)

    def _forward_lsnet(self, x):
        outs = self.net(x)
        return outs[0], outs[1], outs[2]

    def forward(self, x):
        f8, f16, f32 = self._forward_impl(x)
        return {"res3": f8, "res4": f16, "res5": f32}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return self._size_divisibility


# --------------------------------------------------------------------------- #
# 注册：仅骨干 / + BiFPN（与 build_fcos_mobilevigv2_csp_bifpn_backbone 对应）
# --------------------------------------------------------------------------- #
@BACKBONE_REGISTRY.register()
def build_mobile_bb_backbone(cfg, input_shape):
    return MobileBBBackbone(
        model_name=cfg.MODEL.MOBILE_BB.MODEL_NAME,
        weights=cfg.MODEL.MOBILE_BB.WEIGHTS or "",
        input_size=cfg.MODEL.MOBILE_BB.INPUT_SIZE,
    )


@BACKBONE_REGISTRY.register()
def build_fcos_mobile_bb_bifpn_backbone(cfg, input_shape: ShapeSpec):
    from .bifpn import BiFPN

    bottom_up = build_mobile_bb_backbone(cfg, input_shape)
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
        passthrough=cfg.MODEL.BiFPN.PASSTHROUGH,
    )
    return backbone
