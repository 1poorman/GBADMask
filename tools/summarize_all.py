# -*- coding: utf-8 -*-
"""汇总消融实验结果：从各 output/ab-*/log.txt 提取 AP，按配置分组统计均值±std。

用法:
    python tools/summarize_all.py [output_dir_prefix]

输出:
    1. 逐次运行明细
    2. 按配置聚合（多 seed：均值 ± 标准差）
    3. 与 base 的差异及可信度判断（基于配对 seed）
"""
import glob
import os
import re
import sys

import numpy as np

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "output/ab-"


def extract_final_ap(log_path):
    """从 detectron2 训练日志提取最后一次评估的 (bbox AP, segm AP)。"""
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    out = {}
    for task in ("bbox", "segm"):
        pat = (r"copypaste: Task: " + task +
               r"\n.*\n.*copypaste: ([0-9.]+(?:,[0-9.]+){5})")
        ms = re.findall(pat, text)
        if ms:
            v = [float(x) for x in ms[-1].split(",")]
            out[task] = {"AP": v[0], "AP50": v[1], "AP75": v[2],
                         "APs": v[3], "APm": v[4], "APl": v[5]}
    return out or None


def main():
    runs = {}   # config_name -> list of (run_name, segm AP, bbox AP, AP50...)
    for d in sorted(glob.glob(PREFIX + "*")):
        if not os.path.isdir(d):
            continue
        tag = os.path.basename(d)[3:]        # 去掉 "ab-"
        if tag.startswith("_"):
            continue
        log = os.path.join(d, "log.txt")
        if not os.path.isfile(log):
            continue
        r = extract_final_ap(log)
        if not r:
            continue
        # 配置名 = 去掉 _s<seed> 后缀
        cfg_name = re.sub(r"_s\d+$", "", tag)
        seed = tag[len(cfg_name) + 1:] if tag != cfg_name else "-"
        runs.setdefault(cfg_name, []).append(
            (tag, seed, r.get("segm"), r.get("bbox")))

    if not runs:
        print("未找到任何结果（查找前缀 {}）".format(PREFIX))
        return 1

    # ---- 明细 ----
    print("=" * 88)
    print("逐次运行明细")
    print("=" * 88)
    print("%-22s %8s %9s %9s %9s" % ("run", "seed", "segmAP", "AP50", "bboxAP"))
    for cfg in sorted(runs):
        for tag, seed, seg, bb in sorted(runs[cfg]):
            if seg:
                print("%-22s %8s %9.3f %9.3f %9.3f" % (
                    tag, seed, seg["AP"], seg["AP50"], bb["AP"] if bb else 0))

    # ---- 聚合 ----
    print()
    print("=" * 88)
    print("按配置聚合（segm AP）")
    print("=" * 88)
    print("%-16s %5s %9s %9s %9s %9s" % (
        "config", "n", "mean", "std", "min", "max"))
    print("-" * 88)
    agg = {}
    for cfg in sorted(runs):
        aps = [r[2]["AP"] for r in runs[cfg] if r[2]]
        if not aps:
            continue
        a = np.array(aps)
        agg[cfg] = a
        print("%-16s %5d %9.3f %9.3f %9.3f %9.3f" % (
            cfg, len(a), a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0,
            a.min(), a.max()))

    # ---- 配对比较 ----
    print()
    print("=" * 88)
    print("与 base 的配对比较（相同 seed 的差值）")
    print("=" * 88)
    if "base" not in runs:
        print("无 base 组")
        return 0

    def by_seed(cfg):
        return {r[1]: r[2]["AP"] for r in runs.get(cfg, []) if r[2]}

    base_by = by_seed("base")
    for cfg in sorted(runs):
        if cfg == "base":
            continue
        cur = by_seed(cfg)
        pairs = [(s, cur[s], base_by[s]) for s in sorted(
            set(cur) & set(base_by))]
        if not pairs:
            continue
        diffs = np.array([c - b for _, c, b in pairs])
        detail = " ".join(
            "s{}:{:+.3f}".format(s, c - b) for s, c, b in pairs)
        print("%-16s n_pairs=%d  mean_delta=%+.3f  %s" % (
            cfg, len(pairs), diffs.mean(), detail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
