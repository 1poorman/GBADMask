"""把小麦病害分割数据集（HBueHxOW/zzy_dataset，YOLO-seg 格式）转成 COCO instances 格式。

数据源
------
``datasets/HBueHxOW/`` 下有三份内容，只有 ``zzy_dataset`` 带分割标注：

=========================== ==================================================
``zzy_dataset/``            YOLO-seg，images/{train,val,test} + labels/{train,val,test}，905 张图
``zzy_wheat/``              同一批图的 LabelMe 源标注（*.json），仅用于核对类别名
``wheatData_split910/``     纯分类数据（12 类文件夹，7653 张图），**无 mask**，不参与转换
===========================

YOLO-seg 标注格式（每行一个对象）::

    <class_id> <x1> <y1> <x2> <y2> ...

坐标为 **归一化** 到 [0, 1] 的多边形顶点（x, y 交替）。COCO instances 需要
bbox(x, y, w, h)、area、segmentation，均为 **绝对像素**。

类别顺序（YOLO class_id 0~11 → COCO category_id 1~12）
------------------------------------------------------
原始数据里没有 ``classes.txt``，类别名由 ``zzy_wheat/*.json`` 的 LabelMe 标签与
``zzy_dataset/labels/*.txt`` 的 class_id **交叉比对**得出（每类 40~110 张图，
一一对应，无歧义），顺序沿用 YOLO 的 id 顺序。

用法::

    conda activate gbadmask
    cd /home/huachenghao/codes/GBADMask
    python datasets/prepare_wheat_seg_coco.py             # 默认软链接 + 按文件名去重
    python datasets/prepare_wheat_seg_coco.py --copy      # 改为复制图片

两份产物
--------
``wheat_seg``       按**文件名**去重，899 张。仅去掉 6 张跨 split 重名。
``wheat_seg_clean`` 按**文件内容 MD5**去重并移除类别冲突，818 张。无泄漏，推荐用于正式实验::

    python datasets/prepare_wheat_seg_coco.py \
        --out datasets/HBueHxOW/wheat_seg_clean --dedup hash --drop-conflict

两者的差距见 ``DATA.md`` 第 7 节：按文件名去重漏掉了 30 组**异名同图**
（如 ``train/CrownAndRootRot_11.jpg`` 与 ``val/CrownAndRootRot_504.jpg``），
会让 val/test 指标虚高。

输出目录结构（与 README 第 6 节一致，可直接作为 GBADMASK_DATA_ROOT）::

    datasets/HBueHxOW/wheat_seg/
    ├── annotations/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   └── instances_test2017.json     # 测试集，训练脚本不自动注册
    ├── train2017/
    ├── val2017/
    └── test2017/
"""
import argparse
import hashlib
import json
import os
import shutil

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

SRC_ROOT = os.path.join(HERE, "HBueHxOW", "zzy_dataset")
OUT_ROOT = os.path.join(HERE, "HBueHxOW", "wheat_seg")

# 下标 = YOLO class_id，值 = 类别名。COCO category_id = class_id + 1（0 留给背景）
WHEAT_CLASSES = [
    "CrownAndRootRot",    # 0  根冠腐烂
    "HealthyWheat",       # 1  健康小麦
    "LeafRust",           # 2  叶锈病
    "PowderyMildew",      # 3  白粉病
    "WheatLooseSmut",     # 4  散黑穗病
    "WheatAphids",        # 5  蚜虫病
    "WheatCystNematode",  # 6  孢囊线虫病
    "WheatRedSpider",     # 7  红蜘蛛
    "WheatScab",          # 8  赤霉病
    "WheatSharpEyespot",  # 9  纹枯病
    "WheatStalkRot",      # 10 茎基腐
    "WheatTake-all",      # 11 全蚀病
]

# 采集自 AI Studio 公开数据集，README.md 中标注为 CC0
LICENSES = [{"id": 1, "name": "CC0", "url": ""}]
INFO = {
    "description": "Wheat disease instance segmentation (HBueHxOW), converted from YOLO-seg",
    "version": "1.0",
    "year": 2024,
    "contributor": "HBueHxOW",
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_PRIORITY = ["train", "val", "test"]   # 同名图出现在多个 split 时，靠前的优先保留


def polygon_area(seg):
    """shoelace 公式算多边形面积（COCO 的 area 字段语义）。"""
    xs, ys = seg[0::2], seg[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) / 2.0


def parse_yolo_seg_line(line, img_w, img_h):
    """解析一行 YOLO-seg 标注，返回 COCO annotation 的核心字段；非法行返回 None。"""
    parts = line.split()
    if len(parts) < 7:          # cls + 至少 3 个点(x, y) = 7
        return None
    cls_id = int(float(parts[0]))
    if not (0 <= cls_id < len(WHEAT_CLASSES)):
        return None
    coords = [float(v) for v in parts[1:]]
    if len(coords) % 2 != 0:    # 保证 x, y 成对
        coords = coords[:-1]

    seg = []
    for i in range(0, len(coords), 2):
        # 反归一化 + 裁剪到图像范围（部分标注点会略微越界，越界会让 mask 光栅化越界）
        seg.append(min(max(coords[i] * img_w, 0.0), float(img_w)))
        seg.append(min(max(coords[i + 1] * img_h, 0.0), float(img_h)))

    xs, ys = seg[0::2], seg[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    area = polygon_area(seg)
    if area <= 0:
        return None
    return {
        "category_id": cls_id + 1,          # COCO 的 category_id 从 1 开始
        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
        "area": area,
        "segmentation": [seg],
        "iscrowd": 0,
    }


def _file_md5(path, _cache={}):
    """文件内容 MD5（缓存，同一批图会被去重逻辑多次引用）。"""
    if path not in _cache:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _cache[path] = h.hexdigest()
    return _cache[path]


def label_signature(txt_path):
    """标签里出现的**类别集合**，作为"标注是否一致"的判据。

    只看类别、不看多边形顶点：同一张图被重复标注时（本数据有 65 组），
    两份标注的多边形顶点数可能不同，但类别一致，属于"重复"而非"冲突"。
    """
    if not os.path.isfile(txt_path):
        return None
    ids = set()
    for line in open(txt_path):
        line = line.strip()
        if line:
            ids.add(int(float(line.split()[0])))
    return tuple(sorted(ids))


def scan_images():
    """扫描源数据，返回 [(split, 图片路径, 标签路径, 文件名)]，按 split 与文件名排序。"""
    items = []
    for split in SPLIT_PRIORITY:
        img_dir = os.path.join(SRC_ROOT, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for n in sorted(os.listdir(img_dir)):
            if not n.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(n)[0]
            items.append((
                split,
                os.path.join(img_dir, n),
                os.path.join(SRC_ROOT, "labels", split, stem + ".txt"),
                n,
            ))
    return items


def collect_images(dedup="name", drop_conflict=False):
    """按指定口径去重，返回 ``({split: [图片路径]}, 报告 dict)``。

    Args:
        dedup: 去重口径
            * ``none`` 不去重（会得到 905 张，含全部重复）
            * ``name`` 按**文件名**去重（默认，历史行为；只去掉 6 张）
            * ``hash`` 按**文件内容 MD5**去重（推荐；能抓到 75 组重复）
        drop_conflict: 是否丢弃"同一张图被标成不同类别"的组（本数据 6 组）。
                       整组丢弃而非保留其一——无法判断哪个标注是对的。

    同组内保留优先级最高的一个副本：先按 split（train > val > test），
    再按文件名，保证结果**确定可复现**。
    """
    if dedup not in ("none", "name", "hash"):
        raise ValueError("dedup 只能是 none / name / hash，收到 {}".format(dedup))

    items = scan_images()
    # 不去重时每张图自成一组
    if dedup == "none":
        groups = [[it] for it in items]
    else:
        buckets = {}
        for it in items:
            key = os.path.basename(it[1]) if dedup == "name" else _file_md5(it[1])
            buckets.setdefault(key, []).append(it)
        groups = [buckets[k] for k in sorted(buckets, key=str)]

    files = {s: [] for s in SPLIT_PRIORITY}
    report = {
        "scanned": len(items),
        "groups": len(groups),
        "dup_dropped": [],       # 因重复被丢弃的副本
        "conflict_dropped": [],  # 因类别冲突被整组丢弃
    }

    for group in groups:
        # 整组保留顺序：split 优先级 > 文件名
        ordered = sorted(group, key=lambda it: (SPLIT_PRIORITY.index(it[0]), it[3]))

        if drop_conflict and len(group) > 1:
            sigs = {label_signature(it[2]) for it in group}
            if len(sigs) > 1:
                report["conflict_dropped"].append(ordered)
                continue

        files[ordered[0][0]].append(ordered[0][1])
        report["dup_dropped"].extend(it[1] for it in ordered[1:])

    return files, report


def collect_split_files():
    """兼容旧调用：按文件名去重的简化入口。"""
    files, report = collect_images(dedup="name", drop_conflict=False)
    return files, report["dup_dropped"]


def link_or_copy(src, dst, use_copy):
    """在输出目录放置图片：默认软链接（省 458 MB 磁盘），--copy 时改为复制。"""
    if os.path.lexists(dst):
        return
    if use_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def convert_split(split, img_paths, use_copy, out_split_dir):
    os.makedirs(out_split_dir, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    for img_id, src in enumerate(img_paths, start=1):
        base = os.path.basename(src)
        stem = os.path.splitext(base)[0]

        link_or_copy(src, os.path.join(out_split_dir, base), use_copy)

        with Image.open(src) as im:
            w, h = im.size
        images.append({
            "id": img_id, "file_name": base, "width": w, "height": h, "license": 1,
        })

        lbl = os.path.join(SRC_ROOT, "labels", split, stem + ".txt")
        if not os.path.isfile(lbl):
            continue
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ann = parse_yolo_seg_line(line, w, h)
                if ann is None:
                    continue
                ann.update({"id": ann_id, "image_id": img_id})
                annotations.append(ann)
                ann_id += 1

    return images, annotations


def build_categories():
    return [
        {"id": i + 1, "name": name, "supercategory": "wheat_disease"}
        for i, name in enumerate(WHEAT_CLASSES)
    ]


def prepare(out_root=OUT_ROOT, use_copy=False, dedup="name", drop_conflict=False):
    files, report = collect_images(dedup=dedup, drop_conflict=drop_conflict)

    print("扫描 {} 张图，去重口径 '{}' -> {} 组唯一内容"
          .format(report["scanned"], dedup, report["groups"]))
    if report["dup_dropped"]:
        print("  去掉 {} 个重复副本（同组保留 split 优先级最高的一个：train > val > test）"
              .format(len(report["dup_dropped"])))
    if report["conflict_dropped"]:
        print("  丢弃 {} 组**类别冲突**（同一张图被标成不同类别），共 {} 张："
              .format(len(report["conflict_dropped"]),
                      sum(len(g) for g in report["conflict_dropped"])))
        for group in report["conflict_dropped"]:
            names = " | ".join("{}/{}".format(s, n) for s, _, _, n in group)
            sigs = " vs ".join(
                ",".join(WHEAT_CLASSES[i] for i in label_signature(t) or ())
                for _, _, t, _ in group
            )
            print("    {}   ->   {}".format(names, sigs))

    ann_root = os.path.join(out_root, "annotations")
    os.makedirs(ann_root, exist_ok=True)

    categories = build_categories()
    stats = {}
    for split in SPLIT_PRIORITY:
        out_split = "{}2017".format(split)
        images, annotations = convert_split(
            split, files[split], use_copy, os.path.join(out_root, out_split)
        )
        json.dump(
            {
                "info": INFO,
                "licenses": LICENSES,
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            open(os.path.join(ann_root, "instances_{}.json".format(out_split)), "w"),
        )
        stats[split] = (len(images), len(annotations))
        print("  {:<5}: {:>4} 张图, {:>4} 个实例".format(split, *stats[split]))

    print("\n输出目录: {}".format(out_root))
    print("图片合计: {} 张，实例合计 {} 个".format(
        sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())))
    print("类别数: {} （类别 id 连续 1~{}，MODEL.FCOS.NUM_CLASSES 需设为 {}）"
          .format(len(WHEAT_CLASSES), len(WHEAT_CLASSES), len(WHEAT_CLASSES)))
    return out_root


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", action="store_true",
                    help="复制图片到输出目录（默认用相对软链接，省磁盘）")
    ap.add_argument("--out", default=OUT_ROOT, help="输出根目录")
    ap.add_argument("--dedup", default="name", choices=("none", "name", "hash"),
                    help="去重口径：none=不去重 / name=按文件名（默认，历史行为，只去 6 张）/ "
                         "hash=按文件内容 MD5（推荐，能抓到 75 组重复）")
    ap.add_argument("--drop-conflict", action="store_true",
                    help="丢弃同一张图被标成不同类别的组（共 6 组）。整组丢弃，"
                         "因为无法判断哪个标注是对的")
    args = ap.parse_args()

    print("源数据: {}".format(SRC_ROOT))
    prepare(args.out, args.copy, dedup=args.dedup, drop_conflict=args.drop_conflict)
