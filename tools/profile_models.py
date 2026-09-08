# -*- coding: utf-8 -*-
"""统计各候选模型的参数量 / FLOPs(@512) / checkpoint 大小。

从每个实验的 output/<run>/config.yaml（完整 dump）在 CPU 重建模型（不加载权重）：
- 参数量：模型全部参数。
- FLOPs：fvcore FlopCountAnalysis。因 detectron2 输出含 Instances 无法直接 trace，
  用一个只跑到 dense 输出（backbone→neck→FCOS 头→basis，均为 Tensor）的包装模块
  计数；basis(ProtoNet/ProtoNetV2) 的算子无法被逐模块计入（实测影响 < 0.01 G），
  数值以 conv 主导的 backbone/neck/head 为主，跨模型口径一致。
  输入 512x512；FLOPs = multiply-accumulate × 2 的 fvcore 口径。
GPU 延迟/FPS 由 tools/benchmark_fps.py 单独实测（本脚本不碰 GPU）。

用法:
    python tools/profile_models.py            # 全量
    python tools/profile_models.py P0_res     # 只跑某个 run

输出 CSV：output/_figures_m65/stats/model_profile.csv
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn as nn

ROOT = "output"
OUT = os.path.join(ROOT, "_figures_m65", "stats")

RUNS = [
    ("R1_res", "m63e_straw_R1_res", "R50-FPN + ProtoNet + GC"),
    ("P0_res", "m63e_straw_P0_res", "MobileViGv2-M+C3K2 + BiFPN + ProtoNetV2+GC"),
    ("HQ1", "m65_hq1_detail_straw512", "P0_res + res2 detail concat"),
    ("DQ1b", "m65_dq1b_qfl_straw512", "P0_res + QFL"),
    ("BR1_lite", "m65_br1_lite_bw3_straw512", "P0_res + boundary-BCE(lambda3)"),
    ("M2b", "m65_m2b_botres64_straw512", "P0_res, BOTTOM_RES=64"),
    ("M2b_s123", "m65_m2b_s123_straw512", "M2b, seed 123"),
    ("B", "m63d_straw_B", "MobileViGv2-B (capacity, 416-protocol)"),
]


def _tensors(o):
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, (list, tuple)):
        out = []
        for x in o:
            out += _tensors(x)
        return out
    return []


class CounterNet(nn.Module):
    """把模型截到 dense tensor 输出，供 fvcore 计数。"""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.fcos = model.proposal_generator
        self.basis = getattr(model, "basis_module", None)

    def forward(self, t):
        f = self.backbone(t)
        outs = _tensors(self.fcos.forward_head(f, None))
        if self.basis is not None:
            try:
                b, _ = self.basis(f, None)
                outs += _tensors(b.get("bases", []))
            except Exception:
                pass          # 变体 basis 需要训练期输入时忽略，只计骨干+头
        return tuple(outs)


def build(tag):
    sys.path.insert(0, ".")
    from adet.config import get_cfg
    from detectron2.modeling import build_model
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(ROOT, tag, "config.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    model = build_model(cfg)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="?", default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for label, tag, desc in RUNS:
        if args.only and label != args.only:
            continue
        ck = os.path.join(ROOT, tag, "model_final.pth")
        ckpt_mb = os.path.getsize(ck) / 1e6 if os.path.isfile(ck) else None
        rec = dict(run=label, dir=tag, desc=desc,
                   ckpt_MB=round(ckpt_mb, 1) if ckpt_mb else "")
        try:
            model = build(tag)
            n = sum(p.numel() for p in model.parameters())
            rec["params_M"] = round(n / 1e6, 2)
            net = CounterNet(model)
            t = torch.rand(1, 3, 512, 512)
            with torch.no_grad():
                from fvcore.nn import FlopCountAnalysis
                f = FlopCountAnalysis(net, (t,))
            rec["flops_G_512"] = round(f.total() / 1e9, 2)
            print("{} params={}M flops={}G".format(
                label, rec["params_M"], rec["flops_G_512"]), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec["params_M"] = "ERR {}".format(type(e).__name__)
            rec.setdefault("flops_G_512", "")
        rows.append(rec)

    with open(os.path.join(OUT, "model_profile.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "dir", "desc", "params_M",
                                          "flops_G_512", "ckpt_MB"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("saved", os.path.join(OUT, "model_profile.csv"))


if __name__ == "__main__":
    main()
