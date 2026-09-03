# -*- coding: utf-8 -*-
"""从 COCO json 的 categories 中移除语义分割遗留的 ``_background_`` 类。

背景
----
语义分割导出的 COCO 数据（如 PlantVillage 系）常带一个 id=0 的 ``_background_``
占位类。它没有任何实例引用，但会引发两个问题：

1. **metadata 口径冲突**：detectron2 的懒加载 ``load_coco_json`` 会用完整
   categories（含背景）设置 ``thing_classes``，与项目内注册时"过滤背景类"
   的口径（不含）不一致，训练启动即 ``AssertionError``；
2. **语义歧义**：detectron2 的 ``NUM_CLASSES`` 指**前景类数**，把占位背景
   混进 thing 类别会让人误配 ``MODEL.FCOS.NUM_CLASSES``。

本脚本做的事（幂等，重复运行无副作用）：

* 从 categories 中删除名为 ``_background_`` 的条目；
* **其余类别 id 保持不变**（不重编号，避免破坏已有引用）；
* 原 json 备份为同名 ``.bak``（不删除任何文件）；
* 校验没有任何 annotation 引用被删类别，有则拒绝执行。

用法::

    python datasets/clean_background_class.py            # 清洗 datasets/ 下全部
    python datasets/clean_background_class.py Plantv2    # 只清洗指定数据集
"""
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BG_NAMES = {"_background_", "background", "__background__"}


def clean_json(path):
    """返回 (是否修改, 原类别数, 新类别数)。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cats = d.get("categories", [])
    if not cats:
        return False, 0, 0
    bg = [c for c in cats if c.get("name", "").lower() in BG_NAMES]
    if not bg:
        return False, len(cats), len(cats)
    bg_ids = {c["id"] for c in bg}
    refs = [a for a in d.get("annotations", []) if a["category_id"] in bg_ids]
    if refs:
        print("  拒绝: {} 中有 {} 条 annotation 引用了背景类，请先人工处理"
              .format(path, len(refs)))
        return False, len(cats), len(cats)

    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)          # 只备份，不删除原文件
    d["categories"] = [c for c in cats if c["id"] not in bg_ids]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return True, len(cats), len(d["categories"])


def clean_dataset(ds_dir):
    """清洗一个数据集目录下所有 split 的 json。返回处理的 split 数。"""
    n = 0
    for path in sorted(glob.glob(
            os.path.join(ds_dir, "annotations", "instances_*.json"))):
        changed, before, after = clean_json(path)
        split = os.path.basename(path).replace("instances_", "").replace(".json", "")
        if changed:
            print("  {}: {} 类 -> {} 类（原文件备份为 .bak）"
                  .format(split, before, after))
        else:
            print("  {}: 无需修改（{} 类）".format(split, before))
        n += 1
    return n


def main():
    root = HERE
    targets = sys.argv[1:] or sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.isdir(os.path.join(root, d, "annotations")))
    for ds in targets:
        ds_dir = os.path.join(root, ds)
        print("清洗 {}:".format(ds))
        clean_dataset(ds_dir)


if __name__ == "__main__":
    main()
