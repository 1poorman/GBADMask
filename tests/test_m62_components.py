import numpy as np
import torch
from detectron2.structures import BoxMode

# QFL must support continuous positive qualities and propagate finite grads.
from adet.modeling.fcos.fcos_outputs import quality_focal_loss
qfl_logits = torch.zeros(2, 3, requires_grad=True)
qfl_targets = torch.tensor([[0.0, 0.8, 0.0], [0.0, 0.0, 0.0]])
qfl_loss = quality_focal_loss(qfl_logits, qfl_targets, beta=2.0)
qfl_loss.backward()
assert torch.isfinite(qfl_loss) and torch.isfinite(qfl_logits.grad).all()
assert qfl_loss.item() > 0
print("[OK] quality_focal_loss continuous targets + finite grads")

# 1) PSA 模块前向 + 通道守恒
from adet.modeling.blendmask.psa import PSA, Attention
m = PSA(128, 128)
x = torch.randn(2, 128, 80, 80)
y = m(x)
assert y.shape == x.shape, (y.shape, x.shape)
# 低通道分支也可用
m2 = PSA(24, 24)
y2 = m2(torch.randn(1, 24, 64, 64))
assert y2.shape == (1, 24, 64, 64)
print("[OK] PSA forward, shapes:", y.shape, y2.shape)

# 2) build_attention 注册 psa / ema
from adet.modeling.blendmask.basis_module import build_attention
a = build_attention("psa", 128)
print("[OK] build_attention('psa',128) ->", type(a).__name__)

# EMA：24（低层分支）与 128（tower）两档通道都应工作，且通道守恒
from adet.modeling.blendmask.ema import EMA, _ema_groups
for c in (24, 128, 256):
    e = EMA(c, factor=32)
    y = e(torch.randn(2, c, 64, 64))
    assert y.shape == (2, c, 64, 64), (c, y.shape)
    print(f"[OK] EMA(channels={c}) groups={e.groups} -> out {tuple(y.shape)}")
# 分组自适应：24 通道下不应退化到整组（groups 应整除且每组≥4）
assert _ema_groups(24, 32) in (6, 8, 12, 24), _ema_groups(24, 32)
ae = build_attention("ema", 128)
print("[OK] build_attention('ema',128) ->", type(ae).__name__)
print("[OK] _ema_groups(24,32)=", _ema_groups(24, 32), " _ema_groups(128,32)=", _ema_groups(128, 32))

try:
    build_attention("bogus", 8)
except ValueError as e:
    print("[OK] unknown attn raises ValueError")

# 3) 默认配置含新键
from adet.config import get_cfg
cfg = get_cfg()
assert cfg.INPUT.COPYPASTE.ENABLED is False
assert cfg.INPUT.LSJ.ENABLED is False
assert tuple(cfg.INPUT.LSJ.SCALE_RANGE) == (0.3, 1.5)
assert cfg.MODEL.FCOS.CLS_LOSS == "focal"
assert cfg.MODEL.FCOS.QFL_BETA == 2.0
print("[OK] defaults COPYPASTE/LSJ keys present")

# 4) LSJ 变换：随机参数在 get_transform 内生成，作用于 image/coords
from adet.data.augmentation import RandomScaleCrop
aug = RandomScaleCrop(scale_range=(0.5, 1.0), crop_size=(640, 640))
img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
t = aug.get_transform(img)
out_img = t.apply_image(img)
assert out_img.shape == (640, 640, 3), out_img.shape
coords = np.array([[100.0, 100.0], [400.0, 400.0]])
out_c = t.apply_coords(coords)
assert out_c.shape == (2, 2)
print("[OK] RandomScaleCrop applied, out img", out_img.shape, "coords", out_c.shape)

# 5) Copy-Paste 逻辑（用合成 donor 池）
from adet.data.copypaste import apply_copy_paste, DonorPool
bg = np.zeros((100, 100, 3), dtype=np.uint8)
donor_img = np.full((20, 20, 3), 255, dtype=np.uint8)
donor_mask = np.ones((20, 20), dtype=np.uint8)
pool = DonorPool([])
pool.donors = [{"img": donor_img, "mask": donor_mask,
                "polys": [[0, 0, 20, 0, 20, 20, 0, 20]],
                "cw": 20, "ch": 20, "cat_id": 3}]
rng = __import__("random").Random(0)
img2, anns = apply_copy_paste(bg.copy(), [], pool, prob=1.0, max_donors=2, rng=rng)
assert len(anns) == 1, len(anns)
assert anns[0]["bbox_mode"] == BoxMode.XYXY_ABS
# 粘贴位置应有白点
assert img2.sum() > 0
# 只读输入（detectron2 read_image 的 PIL 路径）不应崩溃
ro = bg.copy()
ro.flags.writeable = False
img3, anns3 = apply_copy_paste(ro, [], pool, prob=1.0, max_donors=2,
                               rng=__import__("random").Random(0))
assert img3.flags.writeable and img3.sum() > 0
print("[OK] apply_copy_paste: added", len(anns), "ann, pasted pixels sum =", int(img2.sum()),
      "; read-only input OK")

# 6) PSA 窗口注意力语义：W=28（2 窗口、无 pad）下与手写"列组内全局"一致
for dim, h in [(12, 2), (24, 3), (64, 8)]:
    torch.manual_seed(0)
    a = Attention(dim, h, window=14)
    a.eval()
    H, W = 14, 28
    x = torch.randn(1, dim, H, W)
    with torch.no_grad():
        out = a(x)
        N = H * W
        t = x.flatten(2).transpose(1, 2)
        qkv = a.qkv(t).reshape(1, N, 3, h, dim // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (dim // h) ** -0.5
        idx = torch.arange(N)
        grp = (idx % W) // 14
        m = grp[:, None] == grp[None, :]
        att = att.masked_fill(~m, float("-inf")).softmax(-1)
        ref = (att @ v).transpose(1, 2).reshape(1, N, dim)
        ref = a.proj(ref).transpose(1, 2).reshape(1, dim, H, W)
    d = (out - ref).abs().max().item()
    assert d < 1e-5, (dim, h, d)
    print(f"[OK] PSA window-equivalence dim={dim} h={h} max|diff|={d:.1e}")

# 6b) PSA pad 路径：非整倍数尺寸形状守恒 + 梯度有限
for hw in [(77, 61), (84, 84)]:
    a = Attention(64, 8, window=14)
    x = torch.randn(2, 64, *hw, requires_grad=True)
    y = a(x); y.sum().backward()
    assert y.shape == x.shape and torch.isfinite(y).all(), hw
print("[OK] PSA pad path (77x61, 84x84) shape + finite grads")

# 7) LSJ 纵横比保持（非方形图，64% 的 wheat 训练图为非方形）
from adet.data.augmentation import RandomScaleCrop as _RSC
np.random.seed(0)
_aug = _RSC(scale_range=(0.5, 1.2), crop_size=(320, 320))
_t = _aug.get_transform(np.zeros((506, 800, 3), dtype=np.uint8))
assert _t._new_h / _t._new_w - 506 / 800 < 1e-2, (_t._new_h, _t._new_w)
_o = _t.apply_image(np.zeros((506, 800, 3), dtype=np.uint8))
assert _o.shape == (320, 320, 3), _o.shape
# 大/小缩放两条路径（含混合 crop/pad）
for lo, hi in [(1.0, 1.5), (0.3, 0.5)]:
    np.random.seed(1)
    t2 = _RSC(scale_range=(lo, hi), crop_size=(320, 320)).get_transform(
        np.zeros((506, 800, 3), dtype=np.uint8))
    assert t2.apply_image(np.zeros((506, 800, 3), dtype=np.uint8)).shape == (320, 320, 3)
print("[OK] LSJ aspect preserved (506x800), crop/pad paths OK")

# 8) M6.5 HQ1：BiFPN passthrough + ProtoNetV2 detail 分支
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone as _Backbone
from adet.modeling.backbone.bifpn import BiFPN

class _MockBottomUp(_Backbone):
    """输出 res2-res5 四层（模拟 cspvigv2 的 stride 4/8/16/32）。
    为简化，各层通道统一 32（BiFPN 的 lateral 会按声明自动适配）。"""
    def __init__(self):
        super().__init__()
        self._out_feature_channels = {"res2": 32, "res3": 32, "res4": 32, "res5": 32}
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_features = list(self._out_feature_strides.keys())

    def forward(self, x):
        outs = {}
        for name, stride in self._out_feature_strides.items():
            outs[name] = torch.nn.functional.avg_pool2d(
                x, kernel_size=stride, stride=stride, ceil_mode=True)
        return outs

mock = _MockBottomUp()
fpn = BiFPN(mock, ["res3", "res4", "res5"], out_channels=32,
            num_top_levels=2, num_repeats=2, norm="", passthrough=["res2"])
x_in = torch.randn(1, 32, 64, 64)
out = fpn(x_in)
assert set(out.keys()) == {"p3", "p4", "p5", "p6", "p7", "res2"}, out.keys()
# res2 透传是恒等的：不经过任何 BiFPN 融合节点
assert torch.equal(out["res2"], mock(x_in)["res2"])
shapes = fpn.output_shape()
assert shapes["res2"].channels == 32 and shapes["res2"].stride == 4
try:
    BiFPN(mock, ["res3", "res4", "res5"], 32, 2, 1, passthrough=["p3"])
    raise AssertionError("passthrough 冲突未报错")
except ValueError:
    pass
try:
    BiFPN(mock, ["res3", "res4", "res5"], 32, 2, 1, passthrough=["res9"])
    raise AssertionError("passthrough 缺失未报错")
except ValueError:
    pass
print("[OK] BiFPN passthrough res2 (identity, output_shape, validation)")

# 8b) ProtoNetV2 detail 分支：构建 + 前向 + 关闭时与旧结构 state_dict 兼容
from adet.modeling.blendmask.basis_module2 import ProtoNetV2
from adet.config import get_cfg as _get_cfg

def _basis_cfg(detail_on):
    c = _get_cfg()
    c.MODEL.BASIS_MODULE.NAME = "ProtoNetV2"
    c.MODEL.BASIS_MODULE.DETAIL_ON = detail_on
    c.MODEL.BASIS_MODULE.NORM = "BN"
    c.MODEL.BASIS_MODULE.NUM_CLASSES = 7
    c.MODEL.BASIS_MODULE.LOSS_ON = False
    return c

input_shape = {
    "p3": ShapeSpec(channels=160, stride=8),
    "p4": ShapeSpec(channels=160, stride=16),
    "p5": ShapeSpec(channels=160, stride=32),
    "res2": ShapeSpec(channels=32, stride=4),
}
m_on = ProtoNetV2(_basis_cfg(True), input_shape)
m_off = ProtoNetV2(_basis_cfg(False), input_shape)
assert hasattr(m_on, "detail_refine") and not hasattr(m_off, "detail_refine")
# 关闭时的参数名集合是开启版的子集（旧 checkpoint 兼容）
sd_off = set(m_off.state_dict().keys())
sd_on = set(m_on.state_dict().keys())
assert sd_off < sd_on
assert all("detail_refine" in k or "concat" in k for k in sd_on - sd_off)

feats = {"p3": torch.randn(2, 160, 96, 96), "p4": torch.randn(2, 160, 48, 48),
         "p5": torch.randn(2, 160, 24, 24), "res2": torch.randn(2, 32, 192, 192)}
out_on, _ = m_on(feats)
out_off, _ = m_off({k: v for k, v in feats.items() if k != "res2"})
b_on, b_off = out_on["bases"][0], out_off["bases"][0]
assert b_on.shape == b_off.shape, (b_on.shape, b_off.shape)
# 前向 + 反向梯度有限，且 res2 梯度确实回传到 detail 分支
b_on.sum().backward()
assert all(torch.isfinite(p.grad).all() for p in m_on.parameters() if p.grad is not None)
assert m_on.detail_refine[0].weight.grad is not None
# detail_on 但 res2 缺失时应报清晰错误
try:
    ProtoNetV2(_basis_cfg(True), {k: v for k, v in input_shape.items() if k != "res2"})
    raise AssertionError("DETAIL_SOURCE 缺失未报错")
except ValueError:
    pass
print("[OK] ProtoNetV2 HQ1 detail branch (build/forward/backward/compat)")

# 9) M6.5 BR1-lite：边界加权 mask BCE
from adet.modeling.blendmask.blender import boundary_weighted_mask_loss

# 9a) 边界带几何：8x8 方形 GT（[2:6, 2:6]），3x3 核 → 边界环（周长像素）
gt = torch.zeros(2, 64)
gt.view(2, 8, 8)[:, 2:6, 2:6] = 1.0
logits = torch.zeros(2, 64, requires_grad=True)
out_w3 = boundary_weighted_mask_loss(logits, gt, boundary_weight=3.0)
out_w0 = boundary_weighted_mask_loss(logits, gt, boundary_weight=0.0)
# w=0 严格退化为普通 BCE 均值
ref = torch.nn.functional.binary_cross_entropy_with_logits(
    logits, gt, reduction="none").mean(dim=-1)
assert torch.allclose(out_w0, ref, atol=1e-6)
# 边界像素（(2,2) 属于带内）梯度应比内部像素（(4,4)）大
out_w3.sum().backward()
g = logits.grad.view(2, 8, 8)
assert g[0, 2, 2].abs() > g[0, 4, 4].abs(), (g[0, 2, 2], g[0, 4, 4])
# 量级可比：加权均值不应爆炸（同量级于未加权）
assert out_w3.max().item() < out_w0.max().item() * 5
# 空 mask（全 0 GT）也应有限
empty = torch.zeros(1, 64)
out_e = boundary_weighted_mask_loss(torch.zeros(1, 64), empty, 3.0)
assert torch.isfinite(out_e).all()
# 全 1 mask：dilate−erode 仍得边界环
full = torch.ones(1, 64)
out_f = boundary_weighted_mask_loss(torch.zeros(1, 64), full, 3.0)
assert torch.isfinite(out_f).all()
print("[OK] boundary_weighted_mask_loss (band geometry, w=0 equivalence, grads, edge cases)")

print("ALL SMOKE TESTS PASSED")
