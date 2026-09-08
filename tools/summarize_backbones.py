# -*- coding: utf-8 -*-
"""M6.6 骨干消融汇总：解析两数据集全部臂的最终 segm/bbox AP，
重建各臂模型统计参数量，产出 markdown 表格 + 对比图
（AP 柱状图 + 参数量-AP 散点 Pareto 图）。

用法（队列收队后）:
    python tools/summarize_backbones.py            # 两个数据集都出
    python tools/summarize_backbones.py --ds straw # 只出 Strawberry

产物：stdout 表格 + output/_figures_m66/{table,bar,pareto}_{ds}.[md|png]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, ".")

MARKDOWN_OUT = "output/_figures_m66"

# 臂定义：(展示名, 日志路径, 参数量重建的骨干配置 dict, 数据集, 锚点说明)
# 锚点臂（已有 run）也走同一解析路径，保证口径一致。
VIGV2_STRAW = dict(
    backbone="build_fcos_mobilevigv2_csp_bifpn_backbone",
    vig=dict(version="m", pretrained="weights/MobileViG_V2_M_Class.pth", c3k2=True))


def _arm(display, log, cfg_builder, note=""):
    return dict(display=display, log=log, cfg=cfg_builder, note=note)


def straw_arms():
    def mobile_bb(name, weights):
        return dict(backbone="build_fcos_mobile_bb_bifpn_backbone",
                    mobile_bb=dict(name=name, weights=weights))

    def vigv2(version, weights, c3k2):
        return dict(backbone="build_fcos_mobilevigv2_csp_bifpn_backbone",
                    vig=dict(version=version, pretrained=weights, c3k2=c3k2))

    def r50bifpn():
        return dict(backbone="build_fcos_resnet_bifpn_backbone", r50=True)

    return [
        _arm("R50 + FPN (official R1)", "output/m63e_straw_R1_res/log.txt", None, "锚点"),
        _arm("vigv2-M + C3K2 (P0)", "output/m63e_straw_P0_res/log.txt", VIGV2_STRAW, "锚点"),
        _arm("R50 + BiFPN", "logs/m66_straw_r50bifpn.log", r50bifpn()),
        _arm("vigv2-S + C3K2", "logs/m66_straw_vigv2s.log",
             vigv2("s", "weights/MobileViG_V2_S_Class.pth", True)),
        _arm("MobileViGv2-S", "logs/m66_straw_mvigv2s.log",
             vigv2("s", "weights/MobileViG_V2_S_Class.pth", False)),
        _arm("MobileViGv2-M", "logs/m66_straw_mvigv2m.log",
             vigv2("m", "weights/MobileViG_V2_M_Class.pth", False)),
        _arm("MobileNetV3-L", "logs/m66_straw_mnv3l.log",
             mobile_bb("mnv3_l", "weights/mobilenetv3_large_100_ra_in1k.pth")),
        _arm("MobileNetV4-S", "logs/m66_straw_mnv4s.log",
             mobile_bb("mnv4_s", "weights/mobilenetv4_conv_small_in1k.pth")),
        _arm("LSNet-T", "logs/m66_straw_lsnett.log",
             mobile_bb("lsnet_t", "weights/lsnet_t.pth")),
    ]


def wheat_arms():
    def mobile_bb(name, weights):
        return dict(backbone="build_fcos_mobile_bb_bifpn_backbone",
                    mobile_bb=dict(name=name, weights=weights))

    def vigv2(version, weights, c3k2):
        return dict(backbone="build_fcos_mobilevigv2_csp_bifpn_backbone",
                    vig=dict(version=version, pretrained=weights, c3k2=c3k2))

    def r50bifpn():
        return dict(backbone="build_fcos_resnet_bifpn_backbone", r50=True)

    return [
        _arm("R50 + FPN (official R1)", "output/m63_strat_R1/log.txt", None, "锚点"),
        _arm("vigv2-M + C3K2 (P0)", "output/m63_strat_P0/log.txt", None, "锚点"),
        _arm("R50 + BiFPN", "logs/m66_wheat_r50bifpn.log", r50bifpn()),
        _arm("vigv2-S + C3K2", "logs/m66_wheat_vigv2s.log",
             vigv2("s", "weights/MobileViG_V2_S_Class.pth", True)),
        _arm("MobileViGv2-S", "logs/m66_wheat_mvigv2s.log",
             vigv2("s", "weights/MobileViG_V2_S_Class.pth", False)),
        _arm("MobileViGv2-M", "logs/m66_wheat_mvigv2m.log",
             vigv2("m", "weights/MobileViG_V2_M_Class.pth", False)),
        _arm("MobileNetV3-L", "logs/m66_wheat_mnv3l.log",
             mobile_bb("mnv3_l", "weights/mobilenetv3_large_100_ra_in1k.pth")),
        _arm("MobileNetV4-S", "logs/m66_wheat_mnv4s.log",
             mobile_bb("mnv4_s", "weights/mobilenetv4_conv_small_in1k.pth")),
        _arm("LSNet-T", "logs/m66_wheat_lsnett.log",
             mobile_bb("lsnet_t", "weights/lsnet_t.pth")),
    ]


def parse_final_ap(log_path):
    """取日志中最后一次 eval 的 bbox/segm copypaste 六元组。"""
    if not os.path.exists(log_path):
        return None
    with open(log_path, errors="ignore") as f:
        text = f.read()
    seg = re.findall(
        r"copypaste: Task: segm\n[^\n]*\n[^\n]*copypaste: ([\d.,]+)", text)
    bbx = re.findall(
        r"copypaste: Task: bbox\n[^\n]*\n[^\n]*copypaste: ([\d.,]+)", text)
    if not seg or not bbx:
        return None
    seg = [float(x) for x in seg[-1].split(",")]
    bbx = [float(x) for x in bbx[-1].split(",")]
    return dict(segm=seg, bbox=bbx)


def count_params(cfg_builder, num_classes):
    """按队列同款覆盖重建模型，返回 (骨干参数, 总参数)。锚点臂 cfg=None 时
    用已知实测值。"""
    if cfg_builder is None:
        return None
    import torch
    from adet.config import get_cfg
    from detectron2.modeling import build_model

    cfg = get_cfg()
    cfg.merge_from_file("configs/run-vigv2.yaml")
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.FCOS.NUM_CLASSES = num_classes
    cfg.MODEL.BASIS_MODULE.NUM_CLASSES = num_classes
    cfg.MODEL.BASIS_MODULE.NAME = "ProtoNetV2"
    cfg.MODEL.BASIS_MODULE.ATTN = "gc"
    cfg.MODEL.BASIS_MODULE.LOSS_ON = False
    c = cfg_builder
    cfg.MODEL.BACKBONE.NAME = c["backbone"]
    if "mobile_bb" in c:
        cfg.MODEL.MOBILE_BB.MODEL_NAME = c["mobile_bb"]["name"]
        cfg.MODEL.MOBILE_BB.WEIGHTS = ""
    if "vig" in c:
        cfg.MODEL.VIG.VERSION = c["vig"]["version"]
        cfg.MODEL.VIG.PRETRAINED = ""
        cfg.MODEL.VIG.USE_C3K2 = c["vig"]["c3k2"]
    if c.get("r50"):
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4", "res5"]
    torch.manual_seed(0)
    model = build_model(cfg)
    n_bb = sum(p.numel() for p in model.backbone.bottom_up.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    del model
    return n_bb / 1e6, n_total / 1e6


# 锚点臂的实测参数（冒烟/历史实测；NUM_CLASSES 差异 <0.05M，忽略）
ANCHOR_PARAMS = {
    "R50 + FPN (official R1)": (23.51, 35.36),
    "vigv2-M + C3K2 (P0)": (15.85, 25.96),
}


def collect(ds):
    arms = straw_arms() if ds == "straw" else wheat_arms()
    n_cls = 7 if ds == "straw" else 12
    rows = []
    for a in arms:
        ap = parse_final_ap(a["log"])
        if ap is None:
            rows.append(dict(display=a["display"], note=a["note"], missing=True))
            continue
        if a["cfg"] is not None and a["display"] in ANCHOR_PARAMS:
            nbb, ntot = ANCHOR_PARAMS[a["display"]]
        else:
            pp = count_params(a["cfg"], n_cls)
            if pp is None:
                pp = ANCHOR_PARAMS.get(a["display"], (float("nan"),) * 2)
            nbb, ntot = pp
        rows.append(dict(display=a["display"], note=a["note"],
                         segm=ap["segm"][0], bbox=ap["bbox"][0],
                         aps=ap["segm"][3], ap75=ap["segm"][2],
                         bb_params=nbb, total_params=ntot, missing=False))
    return rows


def emit_table(rows, ds):
    ds_name = "Strawberry (512, 22k, seed42)" if ds == "straw" \
        else "wheat_seg_strat (8k, batch7, seed42)"
    have = [r for r in rows if not r.get("missing")]
    r1 = next((r for r in have if r["display"].startswith("R50 + FPN")), None)
    p0 = next((r for r in have if "P0" in r["display"]), None)
    lines = ["| 骨干 | bb 参数(M) | 总参数(M) | segm AP | bbox AP | APs | AP75 | Δ vs R1 | Δ vs P0 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r.get("missing"):
            lines.append("| {} | — | — | （未出分） | | | | | |".format(r["display"]))
            continue
        d_r1 = "{:+.2f}".format(r["segm"] - r1["segm"]) if r1 else "—"
        d_p0 = "{:+.2f}".format(r["segm"] - p0["segm"]) if p0 else "—"
        lines.append("| {} | {:.2f} | {:.2f} | **{:.2f}** | {:.2f} | {:.1f} | {:.1f} | {} | {} |".format(
            r["display"], r["bb_params"], r["total_params"], r["segm"],
            r["bbox"], r["aps"], r["ap75"], d_r1, d_p0))
    table = "\n".join(lines)
    header = "\n### M6.6 骨干消融（{}）\n\n".format(ds_name)
    return header + table + "\n", have


def emit_figures(have, ds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(MARKDOWN_OUT, exist_ok=True)
    have_sorted = sorted(have, key=lambda r: -r["segm"])
    names = [r["display"] for r in have_sorted]
    aps = [r["segm"] for r in have_sorted]

    # 1) 柱状图
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = []
    for r in have_sorted:
        if r["display"].startswith("R50"):
            colors.append("#8c8c8c")
        elif "P0" in r["display"]:
            colors.append("#d62728")
        else:
            colors.append("#1f77b4")
    bars = ax.barh(range(len(aps)), aps, color=colors)
    ax.set_yticks(range(len(aps)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    for i, (b, v) in enumerate(zip(bars, aps)):
        ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2,
                "{:.2f}".format(v), va="center", fontsize=8)
    ax.set_xlabel("segm AP")
    ax.set_title("Backbone ablation ({})".format(
        "Strawberry 512" if ds == "straw" else "wheat_seg_strat"))
    ax.margins(x=0.12)
    fig.tight_layout()
    p = os.path.join(MARKDOWN_OUT, "bar_{}.png".format(ds))
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # 2) Pareto：总参数 vs AP
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [r["total_params"] for r in have_sorted]
    ys = [r["segm"] for r in have_sorted]
    ax.scatter(xs, ys, s=60, c=colors)
    for r in have_sorted:
        ax.annotate(r["display"], (r["total_params"], r["segm"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    # Pareto 前沿（参数越小 AP 越高为优）
    pts = sorted(zip(xs, ys))
    frontier = []
    best_y = -1e9
    for x, y in pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    fx, fy = zip(*frontier)
    ax.plot(fx, fy, "--", color="#2ca02c", alpha=0.7, label="Pareto frontier")
    ax.set_xlabel("total params (M)")
    ax.set_ylabel("segm AP")
    ax.set_title("Accuracy vs params ({})".format(
        "Strawberry 512" if ds == "straw" else "wheat_seg_strat"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(MARKDOWN_OUT, "pareto_{}.png".format(ds))
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["straw", "wheat", "both"], default="both")
    args = ap.parse_args()
    os.makedirs(MARKDOWN_OUT, exist_ok=True)
    dss = ["straw", "wheat"] if args.ds == "both" else [args.ds]
    for ds in dss:
        rows = collect(ds)
        table, have = emit_table(rows, ds)
        print(table)
        with open(os.path.join(MARKDOWN_OUT, "table_{}.md".format(ds)), "w") as f:
            f.write(table)
        if have:
            p1, p2 = emit_figures(have, ds)
            print("figures: {} {}".format(p1, p2))


if __name__ == "__main__":
    main()
