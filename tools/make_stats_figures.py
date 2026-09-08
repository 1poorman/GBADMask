# -*- coding: utf-8 -*-
"""画模型效率图：参数量 / FLOPs@512 / GPU FPS 三栏条形图。

输入 output/_figures_m65/stats/model_stats.csv；输出同名 PNG。
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join("output", "_figures_m65", "stats")
CSV = os.path.join(OUT, "model_stats.csv")


def main():
    rows = list(csv.DictReader(open(CSV)))
    # 颜色：baseline 灰 / proposed 蓝 / capacity 深蓝 / abl 变体
    FAM = {
        "R1_res": "#7F7F7F",
        "P0_res": "#0072B2",
        "B": "#023fa5",
        "DQ1b": "#56B4E9", "HQ1": "#CC79A7", "M2b": "#D55E00",
        "M2b_s123": "#E69F00", "BR1_lite": "#009E73",
    }
    names = [r["run"] for r in rows]
    colors = [FAM[n] for n in names]
    labels = ["{}\n({})".format(r["run"], r["segm_AP"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    panels = [
        (0, "params_M", "Parameters (M)", None),
        (1, "flops_G_512", "FLOPs @512\\u00d7512 (G)", None),
        (2, "fps_512", "GPU FPS (fp32, batch=1, 512)", None),
    ]
    for ax_i, key, ylab, _ in panels:
        ax = axes[ax_i]
        vals = [float(r[key]) for r in rows]
        bars = ax.bar(range(len(names)), vals, color=colors, width=0.62)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(vals) * 1.18)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, "{:.2f}".format(v),
                    ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Model size / compute / speed \\u00b7 Strawberry \\u00b7 segm AP@final in ( )\n"
        "\\u2014 variants of MobileViGv2-M (P0_res) keep params/FLOPs unchanged; "
        "R50 baseline is smaller-params but higher-FLOPs",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = os.path.join(OUT, "efficiency_params_flops_fps.png")
    fig.savefig(p, dpi=170)
    print("saved", p)


if __name__ == "__main__":
    main()
