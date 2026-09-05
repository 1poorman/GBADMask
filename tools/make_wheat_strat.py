#!/usr/bin/env python
"""生成 wheat_seg_strat：818 图全合并后按类别分层的 85/15 新划分（2026-09-04）。

背景：wheat_seg_clean 的 val(82) 未分层——WCN 仅 1 图 3 实例，稀有类测不出
（AP=0）；且 test 已被用于 4 组模型比较（选择偏差污染），不再适合作"未动用的
最终评估集"。合并 train+val+test 全部 818 图，按每图主类分层重划。

规则：
- 每图主类 = 该图第一个标注的 category_id（本数据集 818/818 图为单类）
- 每类 15% 进 val，但至少 2 图；同 seed=42 可复现
- 图像文件**软链**到新目录（不复制，不改动原 wheat_seg_clean 任何文件）
- 原数据集完全保留：wheat_seg_clean 的 train/val/test 与全部历史结果不受影响

输出：datasets/HBueHxOW/wheat_seg_strat/
    annotations/instances_{train,val}2017.json
    {train,val}2017/  -> 软链到原文件
"""
import json
import os
import random
import shutil
from collections import Counter, defaultdict

SRC = "datasets/HBueHxOW/wheat_seg_clean"
DST = "datasets/HBueHxOW/wheat_seg_strat"
SEED = 42
VAL_FRAC = 0.15
MIN_VAL_IMGS = 2

os.makedirs(os.path.join(DST, "annotations"), exist_ok=True)

img_map = {}  # (split, old_id) -> new_id
imgs_all, anns_all = [], []
nid = 0
for split in ["train", "val", "test"]:
    d = json.load(open(os.path.join(SRC, "annotations", f"instances_{split}2017.json")))
    for im in d["images"]:
        nid += 1
        img_map[(split, im["id"])] = nid
        # 记录源文件相对路径，便于软链
        imgs_all.append(dict(im, id=nid, _src=(split, im["file_name"])))
    for a in d["annotations"]:
        anns_all.append(dict(a, image_id=img_map[(split, a["image_id"])]))

categories = d["categories"]
cat_names = {c["id"]: c["name"] for c in categories}
assert len(imgs_all) == 818 and len(anns_all) == 2121, (len(imgs_all), len(anns_all))

# 每图主类
img_main = {}
for a in anns_all:
    img_main.setdefault(a["image_id"], a["category_id"])
assert len(img_main) == len(imgs_all)  # 本数据集全部图有标注

by_cls = defaultdict(list)
for im in imgs_all:
    by_cls[img_main[im["id"]]].append(im["id"])

random.seed(SEED)
val_ids = set()
for cid, ids in sorted(by_cls.items()):
    ids = ids[:]
    random.shuffle(ids)
    n_val = max(MIN_VAL_IMGS, round(len(ids) * VAL_FRAC))
    val_ids.update(ids[:n_val])

ann_by_img = defaultdict(list)
for a in anns_all:
    ann_by_img[a["image_id"]].append(a)

def dump(split, ids):
    out_img_dir = os.path.join(DST, f"{split}2017")
    os.makedirs(out_img_dir, exist_ok=True)
    ids = sorted(ids)
    imgs, anns = [], []
    for iid in ids:
        im = next(x for x in imgs_all if x["id"] == iid)
        src_split, fname = im["_src"]
        # 软链（已存在则跳过）
        link = os.path.join(out_img_dir, fname)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(SRC, f"{src_split}2017", fname)), link)
        imgs.append({k: v for k, v in im.items() if not k.startswith("_")})
        anns.extend(ann_by_img[iid])
    # 重编 annotation id
    for i, a in enumerate(anns):
        a["id"] = i + 1
    out = {
        "images": imgs, "annotations": anns, "categories": categories,
        "info": {"description": f"wheat_seg_strat {split} (stratified from 818 imgs, seed={SEED})"},
    }
    with open(os.path.join(DST, "annotations", f"instances_{split}2017.json"), "w") as f:
        json.dump(out, f)
    return len(imgs), len(anns)

n_tr, a_tr = dump("train", (im["id"] for im in imgs_all if im["id"] not in val_ids))
n_va, a_va = dump("val", val_ids)
print(f"train: {n_tr} imgs, {a_tr} anns")
print(f"val:   {n_va} imgs, {a_va} anns")

# 验证：每类 val 覆盖
cnt = Counter(a["category_id"] for a in anns_all if a["image_id"] in val_ids)
print("\nval 类别覆盖（实例数）:")
for cid in sorted(cat_names):
    print(f"  {cat_names[cid]:22s} {cnt.get(cid, 0):3d} insts")
print("\n✓ 原数据集未改动（wheat_seg_clean 全部保留）")
