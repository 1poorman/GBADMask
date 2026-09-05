#!/usr/bin/env python
"""COCO 实例分割标注 → YOLO 分割格式（归一化多边形）+ data.yaml。

用法：
    python tools/coco2yolo.py <coco_root> <out_root>
    coco_root 布局：annotations/instances_{train,val}2017.json + {train,val}2017/
输出：
    <out_root>/images/{train,val}/*.jpg   （复制或软链）
    <out_root>/labels/{train,val}/*.txt   （class x1 y1 x2 y2 ... 归一化）
    <out_root>/data.yaml
"""
import json
import os
import shutil
import sys


def convert_split(coco_root, split, out_root):
    ann_path = os.path.join(coco_root, "annotations", f"instances_{split}2017.json")
    img_dir = os.path.join(coco_root, f"{split}2017")
    data = json.load(open(ann_path))

    cat_ids = sorted(c["id"] for c in data["categories"])
    cat_id_map = {cid: i for i, cid in enumerate(cat_ids)}
    cat_names = [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]

    imgs = {im["id"]: im for im in data["images"]}

    out_img = os.path.join(out_root, "images", split)
    out_lbl = os.path.join(out_root, "labels", split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    n_inst, n_img = 0, 0
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        img = imgs.get(ann["image_id"])
        if img is None:
            continue
        w, h = img["width"], img["height"]
        segs = ann["segmentation"]
        if not isinstance(segs, list) or len(segs) == 0:
            continue
        cls = cat_id_map[ann["category_id"]]
        lines = []
        for poly in segs:
            if len(poly) < 6:  # 至少 3 个点
                continue
            xs = poly[0::2]
            ys = poly[1::2]
            coords = []
            for x, y in zip(xs, ys):
                coords.append(f"{max(0.0, min(1.0, x / w)):.6f}")
                coords.append(f"{max(0.0, min(1.0, y / h)):.6f}")
            lines.append(f"{cls} " + " ".join(coords))
        if lines:
            stem = os.path.splitext(img["file_name"])[0]
            with open(os.path.join(out_lbl, stem + ".txt"), "a") as f:
                f.write("\n".join(lines) + "\n")
            n_inst += 1

    # 复制图片（软链会让部分训练器重复扫描，直接复制更稳）
    for img in data["images"]:
        src = os.path.join(img_dir, img["file_name"])
        if os.path.isfile(src):
            dst = os.path.join(out_img, img["file_name"])
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            n_img += 1

    return cat_names, n_img, n_inst


def main():
    coco_root, out_root = sys.argv[1], sys.argv[2]
    all_cats = None
    for split in ["train", "val"]:
        cats, n_img, n_inst = convert_split(coco_root, split, out_root)
        print(f"{split}: {n_img} images, {n_inst} instances, {len(cats)} classes")
        if all_cats is None:
            all_cats = cats
        elif all_cats != cats:
            print(f"WARNING: train/val 类别不一致: {set(all_cats) ^ set(cats)}")

    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_root)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"nc: {len(all_cats)}\n")
        f.write(f"names: {all_cats}\n")
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
