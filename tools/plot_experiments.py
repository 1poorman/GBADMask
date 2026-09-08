# -*- coding: utf-8 -*-
"""绘制高 AP 实验的训练 loss 曲线与评估 segm AP 曲线（GBADMask 结果整理）。

数据源：每个实验 output/<run>/metrics.json（训练期每 ~20 iter 一行，
含 total_loss 等；EVAL_PERIOD=4000 的评估行含 segm/AP）+ output/<run>/log.txt
（训练结束后最后一次评估，iter=MAX_ITER-1，取值口径与论文表一致）。

用法::

    python tools/plot_experiments.py

只绘图/写表，不跑训练、不碰 GPU。输出到 output/_figures_m65/curves/。
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "output"
OUT = os.path.join(ROOT, "_figures_m65", "curves")

# 同一协议（Strawberry val, seed42, 22k, test@512）的 6 个候选
RUNS = [  # (label, run_tag, kind)
    ("P0_res", "m63e_straw_P0_res", "proposed"),
    ("BR1-lite", "m65_br1_lite_bw3_straw512", "abl"),
    ("HQ1", "m65_hq1_detail_straw512", "abl"),
    ("DQ1b", "m65_dq1b_qfl_straw512", "abl"),
    ("M2b", "m65_m2b_botres64_straw512", "abl"),
    ("R1_res (R50)", "m63e_straw_R1_res", "baseline"),
]

# Okabe-Ito (CVD-safe) 分类色；baseline 用灰。
PALETTE = {
    "proposed": "#0072B2",   # blue      - 主模型
    "abl": "#009E73",        # green     - 消融变体
    "baseline": "#7F7F7F",   # gray      - R50 基准（虚线）
}
_ABL_COLORS = ["#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]  # vermillion/red-purple/sky/org


def load_metrics(tag):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, tag, "metrics.json"))]
    train = [r for r in rows if r.get("total_loss") is not None]
    ev = [r for r in rows if "segm/AP" in r]
    return train, ev


def parse_log_final(tag):
    """从 log.txt 提取最后一次评估的 (iter, segmAP)。"""
    text = open(os.path.join(ROOT, tag, "log.txt"), encoding="utf-8",
                errors="replace").read()
    m = re.findall(r"copypaste: Task: segm\n.*\n.*copypaste: ([0-9.]+(?:,[0-9.]+){5})",
                   text)
    segm = [float(x) for x in m[-1].split(",")][0] if m else None
    its = [int(x) for x in re.findall(r"iter: (\d+)", text)]
    return (its[-1] if its else None), segm


def rolling(x, w=41):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def main():
    os.makedirs(OUT, exist_ok=True)
    colors = {}
    for i, (label, tag, kind) in enumerate(RUNS):
        if kind == "abl":
            colors[label] = _ABL_COLORS.pop(0)
        elif kind == "baseline":
            colors[label] = PALETTE["baseline"]
        else:
            colors[label] = PALETTE["proposed"]

    data = {}
    final_txt = {}
    for label, tag, kind in RUNS:
        train, ev = load_metrics(tag)
        fiter, fap = parse_log_final(tag)
        data[label] = dict(train=train, ev=ev, tag=tag,
                           fiter=fiter, fap=fap, kind=kind)
        final_txt[label] = fap

    # ================= 图 1：total loss + segm AP =================
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle(
        "Strawberry val · seed42 · 22k iters · eval@512 · GBADMask M6.3e/M6.5 runs",
        fontsize=13)

    # 按最终 AP 从高到低排列图例顺序（P0_res 主模型第一）
    order = sorted(RUNS, key=lambda r: -(final_txt[r[0]] or 0))
    # 图例顺序 = 固定: 主模型、消融(AP desc)、基准
    order = [r for r in order if r[2] == "proposed"] + \
            [r for r in order if r[2] == "abl"] + \
            [r for r in order if r[2] == "baseline"]

    for label, tag, kind in order:
        d = data[label]
        tr = d["train"]
        it = [r["iteration"] for r in tr]
        loss = [r["total_loss"] for r in tr]
        ls = ":" if kind == "baseline" else "-"
        col = colors[label]
        lw = 2.4 if kind == "proposed" else 1.6
        # 平滑主线
        y = rolling(np.asarray(loss, float))
        axL.plot(np.asarray(it)[len(it) - len(y):], y, color=col, ls=ls, lw=lw,
                 alpha=0.95)
        # AP 曲线（周期评估点 + 训练末最终评估）
        xs = [r["iteration"] for r in d["ev"]]
        ys = [r["segm/AP"] for r in d["ev"]]
        if d["fiter"] is not None and d["fap"] is not None:
            xs = xs + [d["fiter"]]
            ys = ys + [d["fap"]]
        axR.plot(xs, ys, color=col, ls=ls, lw=lw, marker="o", ms=4.5,
                 mfc=col, mec="white", mew=0.8, alpha=0.95)
        axR.annotate("{:.2f}".format(ys[-1]), (xs[-1], ys[-1]),
                     xytext=(6, 4), textcoords="offset points", fontsize=9,
                     color=col, fontweight="bold" if kind == "proposed" else "normal")

    axL.set_xlabel("iteration"); axL.set_ylabel("total loss")
    axL.set_title("(a) Training loss")
    axR.set_xlabel("iteration"); axR.set_ylabel("segm AP  (COCO)")
    axR.set_title("(b) Validation segm AP")
    for ax in (axL, axR):
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = []
    for label, tag, kind in order:
        col = colors[label]
        ls = ":" if kind == "baseline" else "-"
        handles.append(plt.Line2D([0], [0], color=col, ls=ls, lw=2.4 if kind == "proposed" else 1.6,
                                  label="{}  ({:.2f})".format(label, final_txt[label])))
    # 主图例放右上；细线子图例略
    axR.legend(handles=handles, loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, "loss_ap_curves_straw512.png")
    fig.savefig(p, dpi=170)
    print("saved", p)

    # ================= 图 2：P0_res 分项 loss =================
    tr = data["P0_res"]["train"]
    it = np.asarray([r["iteration"] for r in tr])
    fig2, ax = plt.subplots(figsize=(9, 4.6))
    comp = [("loss_mask", "mask BCE"), ("loss_fcos_cls", "fcos cls"),
            ("loss_fcos_loc", "fcos loc"), ("loss_fcos_ctr", "fcos ctr")]
    seq = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for (key, name), c in zip(comp, seq):
        v = np.asarray([r.get(key, 0.0) for r in tr], float)
        y = rolling(v)
        ax.plot(it[len(it) - len(y):], y, color=c, lw=1.8, label=name)
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    ax.set_title("P0_res component losses (smoothed)")
    ax.grid(alpha=0.25, lw=0.5); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, frameon=False)
    fig2.tight_layout()
    p2 = os.path.join(OUT, "loss_components_P0_res.png")
    fig2.savefig(p2, dpi=170)
    print("saved", p2)

    # ================= 配套 CSV：AP 表 =================
    import csv
    with open(os.path.join(OUT, "ap_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "dir", "segm_AP@19999", "segm_AP@final(21999)"])
        for label, tag, kind in RUNS:
            d = data[label]
            ev = d["ev"][-1] if d["ev"] else {}
            w.writerow([label, tag,
                        round(ev.get("segm/AP", float("nan")), 2),
                        round(d["fap"], 2) if d["fap"] else ""])
    print("saved", os.path.join(OUT, "ap_summary.csv"))


if __name__ == "__main__":
    main()
