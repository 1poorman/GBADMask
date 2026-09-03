# 临时脚本：验证 cspvigv2 预训练映射完整性与 C3K2 恒等初始化
import sys
sys.path.insert(0, ".")
import torch
from adet.config import get_cfg
from detectron2.modeling.backbone.build import build_backbone

WEIGHTS = {"ti": "weights/MobileViG_V2_Ti_Class.pth",
           "s": "weights/MobileViG_V2_S_Class.pth",
           "m": "weights/MobileViG_V2_M_Class.pth"}


def build(version, use_c3k2, pretrained):
    cfg = get_cfg()
    cfg.merge_from_file("configs/run-wheat.yaml")
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_list([
        "MODEL.BACKBONE.NAME", "build_fcos_mobilevigv2_csp_bifpn_backbone",
        "MODEL.VIG.VERSION", version,
        "MODEL.VIG.USE_C3K2", str(use_c3k2),
        "MODEL.VIG.PRETRAINED", pretrained,
        "MODEL.RESNETS.OUT_FEATURES", ["res2", "res3", "res4", "res5"],
    ])
    cfg.freeze()
    torch.manual_seed(0)
    return build_backbone(cfg, None)


print("== 映射统计 ==")
for v in ["ti", "s", "m"]:
    bb = build(v, True, "")
    st = bb.bottom_up.backbone.load_mobilevigv2_pretrained(WEIGHTS[v])
    print(f"[{v}] loaded={st['loaded']} skipped={len(st['skipped'])} "
          f"mismatched={len(st['mismatched'])}")
    if st["mismatched"]:
        for m in st["mismatched"][:5]:
            print("   MISMATCH:", m)
    non_head = [k for k in st["skipped"]
                if not (k.startswith("prediction") or k.startswith("head")
                        or k.startswith("dist_head"))]
    print(f"   skipped 中非分类头的键: {len(non_head)}", non_head[:5])

print("\n== C3K2 恒等性验证（同一预训练下，c3k2 开/关输出应一致）==")
x = torch.rand(1, 3, 256, 256)
for v in ["s"]:
    bb_on = build(v, True, WEIGHTS[v])
    bb_off = build(v, False, WEIGHTS[v])
    bb_on.eval()
    bb_off.eval()
    with torch.no_grad():
        o1 = bb_on(x)
        o2 = bb_off(x)
    for k in o1:
        d = (o1[k] - o2[k]).abs().max().item()
        print(f"[{v}] {k}: max|diff| = {d:.2e}")
