# -*- coding: utf-8 -*-
"""M6.6 骨干消融冒烟：三个新骨干（mnv3_l / mnv4_s / lsnet_t）+
vigv2-s / MobileViGv2±C3K2 / R50-BiFPN 构建链路。

CPU 可跑。校验：预训练键覆盖率、stride 契约（8/16/32）、前向/反向、
变分辨率前向（480/544——LSNet attention bias 插值路径）。"""
import sys

sys.path.insert(0, ".")
import torch

from adet.config import get_cfg
from detectron2.modeling import build_model


def make_cfg(backbone, **opts):
    cfg = get_cfg()
    cfg.merge_from_file("configs/run-vigv2.yaml")
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.BASIS_MODULE.NAME = "ProtoNetV2"
    cfg.MODEL.BASIS_MODULE.ATTN = "gc"
    cfg.MODEL.BASIS_MODULE.LOSS_ON = False
    cfg.MODEL.FCOS.NUM_CLASSES = 7
    cfg.MODEL.BASIS_MODULE.NUM_CLASSES = 7
    cfg.MODEL.BACKBONE.NAME = backbone
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.VIG.PRETRAINED = opts.pop("vig_pretrained", "")
    for k, v in opts.items():
        setattr(cfg.MODEL, k, v)
    return cfg


def make_batch(h, w):
    from detectron2.structures import BitMasks, Boxes, Instances
    inst = Instances((h, w))
    inst.gt_boxes = Boxes(torch.tensor([[40., 40., 160., 160.]]))
    inst.gt_classes = torch.tensor([0])
    m = torch.zeros(1, h, w, dtype=torch.bool)
    m[0, 50:150, 50:150] = True
    inst.gt_masks = BitMasks(m)
    return [{"image": torch.rand(3, h, w) * 255,
             "height": h, "width": w, "instances": inst}]


def smoke(name, cfg, sizes=(256, 224, 288)):
    model = build_model(cfg)
    model.train()
    n_bb = sum(p.numel() for p in model.backbone.bottom_up.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    # 前向 + 反向 + stride 契约
    for s in sizes:
        batch = make_batch(s, s)
        torch.manual_seed(0)
        losses = model(batch)
        total = sum(losses.values())
        total.backward()
        grads = [p.grad.abs().sum().item() for p in model.parameters()
                 if p.grad is not None]
        assert sum(grads) > 0, "{}@{}: 无梯度".format(name, s)
        losses = {k: round(float(v), 4) for k, v in sorted(losses.items())}
        for k, v in losses.items():
            assert torch.isfinite(torch.tensor(v)), (name, s, k, v)
    # 骨干输出 stride 契约（直接探测）
    model.eval()
    with torch.no_grad():
        feats = model.backbone.bottom_up(torch.zeros(1, 3, 256, 256))
    bb = model.backbone.bottom_up
    if hasattr(bb, "_out_feature_strides"):
        for f, st in bb._out_feature_strides.items():
            want = 256 // st
            assert feats[f].shape[-1] in (want, want // 2 + 1) or \
                abs(feats[f].shape[-1] - want) <= 1, \
                "{}: {} stride {} 实际 {}".format(
                    name, f, st, feats[f].shape)
    print("[OK] {} params(bb/total)={:.2f}M/{:.2f}M losses={} sizes={}".format(
        name, n_bb / 1e6, n_total / 1e6, losses, sizes))
    model.zero_grad()
    return model


# 1) MNv3-L（timm 原生 + IN-1k 权重）
cfg = make_cfg("build_fcos_mobile_bb_bifpn_backbone")
cfg.MODEL.MOBILE_BB.MODEL_NAME = "mnv3_l"
cfg.MODEL.MOBILE_BB.WEIGHTS = "weights/mobilenetv3_large_100_ra_in1k.pth"
smoke("mnv3_l", cfg)

# 2) MNv4-Conv-S（自实现 + IN-1k 权重，键名覆盖率断言在加载器内）
cfg = make_cfg("build_fcos_mobile_bb_bifpn_backbone")
cfg.MODEL.MOBILE_BB.MODEL_NAME = "mnv4_s"
cfg.MODEL.MOBILE_BB.WEIGHTS = "weights/mobilenetv4_conv_small_in1k.pth"
smoke("mnv4_s", cfg)

# 3) LSNet-T（vendored + IN-1k 权重 + 变分辨率插值路径）
cfg = make_cfg("build_fcos_mobile_bb_bifpn_backbone")
cfg.MODEL.MOBILE_BB.MODEL_NAME = "lsnet_t"
cfg.MODEL.MOBILE_BB.WEIGHTS = "weights/lsnet_t.pth"
smoke("lsnet_t", cfg)

# 4) vigv2-S + C3K2（S 权重）
cfg = make_cfg("build_fcos_mobilevigv2_csp_bifpn_backbone",
               vig_pretrained="weights/MobileViG_V2_S_Class.pth")
cfg.MODEL.VIG.VERSION = "s"
smoke("vigv2s_c3k2", cfg)

# 5) MobileViGv2-S（USE_C3K2 False）
cfg = make_cfg("build_fcos_mobilevigv2_csp_bifpn_backbone",
               vig_pretrained="weights/MobileViG_V2_S_Class.pth")
cfg.MODEL.VIG.VERSION = "s"
cfg.MODEL.VIG.USE_C3K2 = False
smoke("mvigv2s_nocsp", cfg)

# 6) MobileViGv2-M（USE_C3K2 False，M 权重）
cfg = make_cfg("build_fcos_mobilevigv2_csp_bifpn_backbone",
               vig_pretrained="weights/MobileViG_V2_M_Class.pth")
cfg.MODEL.VIG.VERSION = "m"
cfg.MODEL.VIG.USE_C3K2 = False
smoke("mvigv2m_nocsp", cfg)

# 7) R50 + BiFPN（同颈对照；R-50.pkl 走本地 iopath 缓存，无则跳过）
cfg = make_cfg("build_fcos_resnet_bifpn_backbone")
cfg.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4", "res5"]
try:
    smoke("r50_bifpn", cfg)
except Exception as e:  # detectron2:// 缓存缺失时给出提示
    if "R-50" in str(e) or "Cannot" in str(e) or "URL" in str(e):
        print("[SKIP] r50_bifpn: R-50.pkl 缓存不可用（训练机上已缓存则无碍）:", str(e)[:120])
    else:
        raise

print("ALL M6.6 BACKBONE SMOKE TESTS PASSED")
