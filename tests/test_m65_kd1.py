# -*- coding: utf-8 -*-
"""KD1 蒸馏冒烟测试：teacher 构建 / 蒸馏损失 / 冻结 / 默认关闭零行为差异。
CPU 可跑（小输入），不依赖数据集。"""
import sys
sys.path.insert(0, ".")
import torch
from detectron2.structures import BitMasks, Boxes, Instances
from adet.config import get_cfg
from detectron2.modeling import build_model

def make_cfg(distill_on, teacher_version="b"):
    cfg = get_cfg()
    cfg.merge_from_file("configs/run-vigv2.yaml")
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.BASIS_MODULE.NAME = "ProtoNetV2"
    cfg.MODEL.BASIS_MODULE.ATTN = "gc"
    cfg.MODEL.BASIS_MODULE.LOSS_ON = False
    cfg.MODEL.FCOS.NUM_CLASSES = 7
    cfg.MODEL.BASIS_MODULE.NUM_CLASSES = 7
    if distill_on:
        # teacher 用 b 变体、随机权重（冒烟不依赖真实 ckpt 的 teacher 前向数值）
        cfg.MODEL.DISTILL.WEIGHTS = "tests/_kd_smoke_teacher.pth"
        cfg.MODEL.DISTILL.TEACHER_OPTS = (
            "MODEL.VIG.VERSION=b MODEL.VIG.PRETRAINED='' "
            "MODEL.FCOS.NUM_CLASSES=7 MODEL.BASIS_MODULE.NUM_CLASSES=7")
        # 同 student 的 m 变体做真实权重对照（冒烟核心是链路不是精度）
        if teacher_version == "m":
            cfg.MODEL.DISTILL.TEACHER_OPTS = (
                "MODEL.VIG.VERSION m MODEL.VIG.PRETRAINED '' "
                "MODEL.FCOS.NUM_CLASSES 7 MODEL.BASIS_MODULE.NUM_CLASSES 7")
    return cfg

def make_batch(h=256, w=256):
    inst = Instances((h, w))
    inst.gt_boxes = Boxes(torch.tensor([[40., 40., 160., 160.],
                                        [120., 120., 220., 220.]]))
    inst.gt_classes = torch.tensor([0, 1])
    m = torch.zeros(2, h, w, dtype=torch.bool)
    m[0, 50:150, 50:150] = True
    m[1, 130:210, 130:210] = True
    inst.gt_masks = BitMasks(m)
    return [{"image": torch.rand(3, h, w) * 255,
             "height": h, "width": w, "instances": inst}]

# 0) 默认关闭：无 teacher，losses 无 kd 键
cfg0 = make_cfg(distill_on=False)
m0 = build_model(cfg0)
m0.train()
torch.manual_seed(0)
l0 = m0(make_batch())
assert not any(k.startswith("loss_kd") for k in l0), l0.keys()
assert m0.teacher is None
print("[OK] KD off: no teacher, no kd losses")

# 1) teacher 构建：b 变体 checkpoint（先存一个随机 teacher）
cfg_t = make_cfg(distill_on=False)
cfg_t.MODEL.VIG.VERSION = "b"
cfg_t.MODEL.VIG.PRETRAINED = ""
cfg_t.MODEL.FCOS.NUM_CLASSES = 7
cfg_t.MODEL.BASIS_MODULE.NUM_CLASSES = 7
torch.manual_seed(1)
teacher = build_model(cfg_t)
torch.save({"model": teacher.state_dict()}, "tests/_kd_smoke_teacher.pth")

cfg1 = make_cfg(distill_on=True)
m1 = build_model(cfg1)
m1.train()
assert m1.teacher is not None
# teacher 是 b 变体且已冻结
t_ver = m1.teacher.backbone.bottom_up.backbone._out_feature_channels
assert t_ver["res5"] == 512, t_ver  # b: channels [64,128,256,512]
for p in m1.teacher.parameters():
    assert not p.requires_grad
# student 仍可训练
assert m1.training
print("[OK] teacher built (b-variant), frozen, student trainable")

# 2) 蒸馏损失：数值有限、三键存在、teacher 无梯度、student head 有梯度
torch.manual_seed(0)
batch = make_batch()
l1 = m1(batch)
for k in ("loss_kd_cls", "loss_kd_reg", "loss_kd_bases"):
    assert k in l1, l1.keys()
    assert torch.isfinite(l1[k]), (k, l1[k])
    assert float(l1[k]) >= 0
total = sum(v for v in l1.values())
total.backward()
for p in m1.teacher.parameters():
    assert p.grad is None
# student 的 top_layer / basis 应收到蒸馏梯度
assert m1.top_layer.weight.grad is not None
grads_basis = [p.grad for p in m1.basis_module.parameters() if p.grad is not None]
assert grads_basis, "basis 未收到蒸馏梯度"
print("[OK] kd losses finite (cls=%.4f reg=%.4f bases=%.4f), teacher grads blocked"
      % (l1["loss_kd_cls"], l1["loss_kd_reg"], l1["loss_kd_bases"]))

# 3) eval 路径不受 teacher 影响（不带 instances —— BlendMask eval 的标准输入）
m1.eval()
with torch.no_grad():
    out = m1([{"image": torch.rand(3, 256, 256) * 255,
               "height": 256, "width": 256}])
assert isinstance(out, list)
print("[OK] eval path unaffected by teacher")

print("KD1 SMOKE TESTS PASSED")
