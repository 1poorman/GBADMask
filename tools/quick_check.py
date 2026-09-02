# -*- coding: utf-8 -*-
"""快速验证一组配置能否构建、前向、反向（CPU，随机数据，不需要数据集）。

用法:
    python tools/quick_check.py "MODEL.VIG.CSP_STYLE c2f" "SEED 42" ...
"""
import sys
import time

import torch

sys.path.insert(0, ".")
from adet.config import get_cfg
from detectron2.modeling import build_model


def check(opts):
    cfg = get_cfg()
    cfg.merge_from_file("configs/run-wheat.yaml")
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.BASIS_MODULE.NAME = "ProtoNetV2"
    cfg.MODEL.BASIS_MODULE.ATTN = "gc"
    cfg.MODEL.BASIS_MODULE.LOSS_ON = True
    cfg.merge_from_list(opts)
    cfg.freeze()
    torch.manual_seed(0)
    model = build_model(cfg)
    model.train()
    n = sum(p.numel() for p in model.parameters())

    # 构造带 GT 的输入（含在线 basis_sem）
    from detectron2.structures import BitMasks, Boxes, Instances
    h = w = 256
    inst = Instances((h, w))
    inst.gt_boxes = Boxes(torch.tensor([[40, 40, 160, 160],
                                        [120, 120, 220, 220]],
                                       dtype=torch.float32))
    inst.gt_classes = torch.tensor([0, 1])
    m = torch.zeros(2, h, w, dtype=torch.bool)
    m[0, 50:150, 50:150] = True
    m[1, 130:210, 130:210] = True
    inst.gt_masks = BitMasks(m)
    basis_sem = torch.zeros(h, w, dtype=torch.long)
    basis_sem[50:150, 50:150] = 1
    basis_sem[130:210, 130:210] = 2

    batched = [{
        "image": torch.rand(3, h, w) * 255,
        "height": h, "width": w,
        "instances": inst, "basis_sem": basis_sem,
    }]
    losses = model(batched)
    total = sum(v for v in losses.values())
    total.backward()
    has_sem = "loss_basis_sem" in losses
    return n, has_sem, float(total)


def main():
    cases = [
        ("c2f         ", ["MODEL.VIG.CSP_STYLE", "c2f"]),
        ("c2f+c3k5    ", ["MODEL.VIG.CSP_STYLE", "c2f", "MODEL.VIG.MID_KERNEL", "5"]),
        ("c3k7        ", ["MODEL.VIG.CSP_STYLE", "c2f", "MODEL.VIG.MID_KERNEL", "7"]),
        ("dysample    ", ["MODEL.BiFPN.UPSAMPLE", "dysample"]),
        ("eca         ", ["MODEL.BiFPN.ATTN", "eca"]),
        ("dy+eca      ", ["MODEL.BiFPN.UPSAMPLE", "dysample",
                        "MODEL.BiFPN.ATTN", "eca"]),
        ("ft+detach   ", ["MODEL.BASIS_MODULE.LOSS_ON", "True",
                         "MODEL.BASIS_MODULE.SEM_LOSS", "focal_tversky",
                         "MODEL.BASIS_MODULE.SEM_DETACH", "True",
                         "MODEL.BASIS_MODULE.LOSS_WEIGHT", "0.05"]),
        ("unified+det ", ["MODEL.BASIS_MODULE.LOSS_ON", "True",
                          "MODEL.BASIS_MODULE.SEM_LOSS", "unified_focal",
                          "MODEL.BASIS_MODULE.SEM_DETACH", "True",
                          "MODEL.BASIS_MODULE.LOSS_WEIGHT", "0.05"]),
        ("ft+nodetach ", ["MODEL.BASIS_MODULE.LOSS_ON", "True",
                          "MODEL.BASIS_MODULE.SEM_LOSS", "focal_tversky",
                          "MODEL.BASIS_MODULE.SEM_DETACH", "False",
                          "MODEL.BASIS_MODULE.LOSS_WEIGHT", "0.05"]),
    ]
    print("%-14s %10s %8s %9s" % ("config", "params(M)", "sem_loss", "total"))
    for name, opts in cases:
        try:
            t0 = time.time()
            n, has_sem, total = check(opts)
            print("%-14s %10.2f %8s %9.4f  (%.0fs)" % (
                name, n / 1e6, "yes" if has_sem else "no", total,
                time.time() - t0))
        except Exception as e:
            print("%-14s FAIL: %s: %s" % (name, type(e).__name__, e))


if __name__ == "__main__":
    main()
