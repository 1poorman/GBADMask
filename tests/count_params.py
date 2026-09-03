# 临时脚本：统计各候选旗舰配置的参数量（CPU 构建，不加载预训练权重）
import sys
sys.path.insert(0, ".")
import torch
from adet.config import get_cfg
from detectron2.modeling import build_model


def count(cfg_file, opts):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_list(opts)
    cfg.freeze()
    torch.manual_seed(0)
    model = build_model(cfg)
    return sum(p.numel() for p in model.parameters()) / 1e6


VIG2 = "build_fcos_mobilevigv2_csp_bifpn_backbone"
cases = [
    ("r50-protonet (基准)", "configs/run-wheat-r50.yaml",
     ["MODEL.BASIS_MODULE.NAME", "ProtoNet"]),
    ("r50-v2", "configs/run-wheat-r50.yaml",
     ["MODEL.BASIS_MODULE.NAME", "ProtoNetV2"]),
    ("r101-v2", "configs/run-wheat-r50.yaml",
     ["MODEL.BASIS_MODULE.NAME", "ProtoNetV2", "MODEL.RESNETS.DEPTH", "101"]),
    ("vigv2-ti+c3k2", "configs/run-wheat.yaml",
     ["MODEL.BACKBONE.NAME", VIG2, "MODEL.VIG.VERSION", "ti",
      "MODEL.RESNETS.OUT_FEATURES", ["res2", "res3", "res4", "res5"]]),
    ("vigv2-s+c3k2", "configs/run-wheat.yaml",
     ["MODEL.BACKBONE.NAME", VIG2, "MODEL.VIG.VERSION", "s",
      "MODEL.RESNETS.OUT_FEATURES", ["res2", "res3", "res4", "res5"]]),
    ("vigv2-s 无c3k2", "configs/run-wheat.yaml",
     ["MODEL.BACKBONE.NAME", VIG2, "MODEL.VIG.VERSION", "s",
      "MODEL.VIG.USE_C3K2", "False",
      "MODEL.RESNETS.OUT_FEATURES", ["res2", "res3", "res4", "res5"]]),
    ("vigv2-m+c3k2", "configs/run-wheat.yaml",
     ["MODEL.BACKBONE.NAME", VIG2, "MODEL.VIG.VERSION", "m",
      "MODEL.RESNETS.OUT_FEATURES", ["res2", "res3", "res4", "res5"]]),
    ("cspvig(旧,v2)", "configs/run-wheat.yaml",
     ["MODEL.BASIS_MODULE.NAME", "ProtoNetV2"]),
]

for name, f, opts in cases:
    try:
        n = count(f, opts)
        print("%-22s %8.2f M" % (name, n))
    except Exception as e:
        print("%-22s FAIL: %s: %s" % (name, type(e).__name__, e))
