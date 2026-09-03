import numpy as np
import torch
from detectron2.structures import BoxMode

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
print("[OK] apply_copy_paste: added", len(anns), "ann, pasted pixels sum =", int(img2.sum()))

print("ALL SMOKE TESTS PASSED")
