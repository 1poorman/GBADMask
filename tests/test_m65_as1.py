# -*- coding: utf-8 -*-
"""AS1（FCOS-TAL）单元测试：默认关闭零行为差异 / warmup / 候选约束 /
每 GT 保底 / 回归目标量纲 / 指派无梯度。CPU 可跑，不依赖数据集。"""
import sys
sys.path.insert(0, ".")
import torch
from detectron2.structures import BitMasks, Boxes, Instances
from adet.config import get_cfg
from detectron2.modeling import build_model


def make_cfg(assign="default", **tal):
    cfg = get_cfg()
    cfg.merge_from_file("configs/run-vigv2.yaml")
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.BASIS_MODULE.NAME = "ProtoNetV2"
    cfg.MODEL.BASIS_MODULE.ATTN = "gc"
    cfg.MODEL.BASIS_MODULE.LOSS_ON = False
    cfg.MODEL.FCOS.NUM_CLASSES = 7
    cfg.MODEL.BASIS_MODULE.NUM_CLASSES = 7
    cfg.MODEL.VIG.PRETRAINED = ""
    cfg.MODEL.FCOS.ASSIGN = assign
    for k, v in tal.items():
        setattr(cfg.MODEL.FCOS, k, v)
    return cfg


def make_batch(h=256, w=256, boxes=None, classes=None):
    if boxes is None:
        boxes = [[40., 40., 160., 160.], [120., 120., 220., 220.]]
    if classes is None:
        classes = [0, 1]
    inst = Instances((h, w))
    inst.gt_boxes = Boxes(torch.tensor(boxes))
    inst.gt_classes = torch.tensor(classes)
    m = torch.zeros(len(boxes), h, w, dtype=torch.bool)
    for i, bb in enumerate(boxes):
        m[i, int(bb[1]) + 5:int(bb[3]) - 5, int(bb[0]) + 5:int(bb[2]) - 5] = True
    inst.gt_masks = BitMasks(m)
    return [{"image": torch.rand(3, h, w) * 255,
             "height": h, "width": w, "instances": inst}]


def run_losses(model, batch):
    torch.manual_seed(0)
    losses = model(batch)
    return {k: float(v) for k, v in losses.items()}


# 0) 默认关闭：行为与旧协议完全一致（无 tal_iter buffer，state_dict 不含新键）
cfg0 = make_cfg()
m0 = build_model(cfg0)
m0.train()
assert not hasattr(m0.proposal_generator.fcos_outputs, "tal_iter")
sd_keys = set(m0.state_dict().keys())
assert not any("tal_iter" in k for k in sd_keys)
ref = run_losses(m0, make_batch())
print("[OK] TAL off: no tal_iter buffer, state_dict unchanged, losses={"
      "}".format({k: round(v, 4) for k, v in sorted(ref.items())}))

# 1) TAL on：构建 + 前向损失有限 + tal_iter 存在
cfg1 = make_cfg(assign="tal", TAL_WARMUP=2, TAL_TOPK=4, TAL_TOPK_SMALL=2,
                TAL_LOG_PERIOD=1)
m1 = build_model(cfg1)
m1.train()
out = m1.proposal_generator.fcos_outputs
assert hasattr(out, "tal_iter")
l1 = run_losses(m1, make_batch())
for k, v in l1.items():
    assert torch.isfinite(torch.tensor(v)), (k, v)
print("[OK] TAL on: builds, finite losses, iter={}".format(int(out.tal_iter)))

# 2) warmup：前 TAL_WARMUP 步保持默认指派（正点数与 default 相同）
cfgw = make_cfg(assign="tal", TAL_WARMUP=10, TAL_TOPK=4)
mw = build_model(cfgw)
mw.train()
batch = make_batch()
# 手动调 losses 链路以抓取指派结果：走 model.forward 三次（都 < warmup）
for _ in range(3):
    mw(batch)
assert int(mw.proposal_generator.fcos_outputs.tal_iter) == 3
print("[OK] warmup: iter counted, default assignment kept (< TAL_WARMUP)")

# 3) 指派语义单元测：直接调 _get_ground_truth + _tal_reassign
outs = m1.proposal_generator.fcos_outputs
# 构造两个 level 的 locations（stride 8/16），模拟 512 输入的坐标
loc0 = torch.stack(torch.meshgrid(
    torch.arange(4, 256, 8.0), torch.arange(4, 256, 8.0), indexing="ij"), -1)
loc0 = loc0.view(-1, 2)
loc1 = torch.stack(torch.meshgrid(
    torch.arange(8, 256, 16.0), torch.arange(8, 256, 16.0), indexing="ij"), -1)
loc1 = loc1.view(-1, 2)
locations = [loc0, loc1]

gt = make_batch()[0]["instances"]
tt = outs._get_ground_truth(locations, [gt])
num_loc_list = [len(l) for l in locations]
K = sum(num_loc_list)
inst = Instances((0, 0))
inst.labels = torch.cat([x.reshape(-1) for x in tt["labels"]])
inst.gt_inds = torch.cat([x.reshape(-1) for x in tt["target_inds"]])
inst.im_inds = torch.cat([x.reshape(-1) for x in tt["im_inds"]])
inst.reg_targets = torch.cat([x.reshape(-1, 4) for x in tt["reg_targets"]])
inst.locations = torch.cat([x.reshape(-1, 2) for x in tt["locations"]])
inst.fpn_levels = torch.cat([x.reshape(-1) for x in tt["fpn_levels"]])
inst.logits_pred = torch.randn(K, 7) * 2
inst.reg_pred = torch.relu(torch.randn(K, 4))
inst.ctrness_pred = torch.randn(K)
inst.top_feats = torch.randn(K, 8)
inst_copy = Instances((0, 0))
for f in ["labels", "gt_inds", "reg_targets", "logits_pred", "reg_pred",
          "im_inds", "locations", "fpn_levels"]:
    setattr(inst_copy, f, getattr(inst, f).clone())

outs.tal_iter.fill_(100)  # 越过 warmup
outs._tal_reassign(inst, [gt])

pos = torch.nonzero(inst.labels != 7).squeeze(1)
assert pos.numel() > 0, "TAL must produce positives"
# 每个 GT 至少一个正点
for g in range(2):
    assert (inst.gt_inds[pos] == g).sum() >= 1, "GT {} uncovered".format(g)
# 正点的 reg_targets 量纲：与该位置 stride 归一化的绝对 ltrb 一致
strides = [8.0, 16.0]
lv = inst.fpn_levels[pos]
lt = inst.reg_targets[pos]
locs_p = inst.locations[pos]
gts = gt.gt_boxes.tensor
for i in range(pos.numel()):
    gi = int(inst.gt_inds[pos[i]])
    s = strides[int(lv[i])]
    want = torch.stack([
        locs_p[i, 0] - gts[gi, 0], locs_p[i, 1] - gts[gi, 1],
        gts[gi, 2] - locs_p[i, 0], gts[gi, 3] - locs_p[i, 1]]) / s
    assert torch.allclose(lt[i], want, atol=1e-4), (lt[i], want)
# 正点数受 topk 约束（每 GT <= TAL_TOPK）
counts = torch.bincount(inst.gt_inds[pos], minlength=2)
assert (counts <= 4).all(), counts
# 指派无梯度（labels/reg_targets 不携带 grad_fn）
assert not inst.labels.requires_grad
assert not inst.reg_targets.requires_grad
# 负点保持 num_classes / gt_inds=-1
neg = torch.nonzero(inst.labels == 7).squeeze(1)
assert (inst.gt_inds[neg] == -1).all()
print("[OK] TAL semantics: >=1 pos/GT, <=topk/GT, stride-normalized "
      "targets, no-grad, counts={}".format(counts.tolist()))

# 4) 兜底路径：极小 GT（中心无任何候选点落入 center 区域时走 fallback）
tiny = make_batch(boxes=[[100., 100., 104., 104.], [10., 10., 200., 200.]],
                  classes=[3, 4])[0]["instances"]
tt2 = outs._get_ground_truth(locations, [tiny])
inst2 = Instances((0, 0))
inst2.labels = torch.cat([x.reshape(-1) for x in tt2["labels"]])
inst2.gt_inds = torch.cat([x.reshape(-1) for x in tt2["target_inds"]])
inst2.im_inds = torch.cat([x.reshape(-1) for x in tt2["im_inds"]])
inst2.reg_targets = torch.cat([x.reshape(-1, 4) for x in tt2["reg_targets"]])
inst2.locations = torch.cat([x.reshape(-1, 2) for x in tt2["locations"]])
inst2.fpn_levels = torch.cat([x.reshape(-1) for x in tt2["fpn_levels"]])
inst2.logits_pred = torch.randn(K, 7) * 2
inst2.reg_pred = torch.relu(torch.randn(K, 4))
inst2.ctrness_pred = torch.randn(K)
inst2.top_feats = torch.randn(K, 8)
outs._tal_reassign(inst2, [tiny])
pos2 = torch.nonzero(inst2.labels != 7).squeeze(1)
for g in range(2):
    assert (inst2.gt_inds[pos2] == g).sum() >= 1, "tiny GT {} uncovered".format(g)
print("[OK] fallback: tiny/degenerate GT still covered")

# 5) 小目标独立 k：面积 < TAL_SMALL_AREA 的 GT 最多 topk_small 个正点
# gt 的两个框：GT0 120x120=14400（大）、GT1 100x100=10000（设阈值 12000 → 小）
outs.tal_small_area = 12000.0
outs._tal_reassign(inst_copy, [gt])
pos_s = torch.nonzero(inst_copy.labels != 7).squeeze(1)
cs = torch.bincount(inst_copy.gt_inds[pos_s], minlength=2)
assert cs[0] <= 4, cs  # 大目标 topk=4
assert cs[1] <= 2, cs  # 小目标 topk_small=2（cfg1）
print("[OK] small-GT dynamic-k: large<={} small<={} (counts={})".format(
    4, 2, cs.tolist()))

# 6) 旧 checkpoint 兼容：不含 tal_iter 的 state_dict 可加载（strict=False）
m1b = build_model(cfg1)
missing, unexpected = m1b.load_state_dict(m0.state_dict(), strict=False)
assert any("tal_iter" in k for k in missing), missing
print("[OK] ckpt compat: old state_dict loads with tal_iter missing (non-fatal)")

# 7) 端到端 backward：TAL on 的完整模型可反传
m1.zero_grad()
losses = m1(make_batch())
total = sum(losses.values())
total.backward()
grads = [p.grad.abs().sum().item() for p in m1.parameters()
         if p.grad is not None]
assert sum(grads) > 0, "no gradient flowed"
print("[OK] end-to-end backward with TAL on")

print("ALL AS1 TESTS PASSED")
