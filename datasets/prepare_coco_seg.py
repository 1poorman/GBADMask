"""把 Ultralytics 的 YOLO-seg 数据集转换成 COCO instances 格式。

用途
----
本项目需要 COCO 格式的实例分割标注（`gt_masks`），而网络受限环境下能拿到的小
数据集（coco128-seg / coco8-seg）是 YOLO-seg 格式，本脚本负责转换。

YOLO-seg 标注格式（每行一个对象）::

    <class_id> <x1> <y1> <x2> <y2> ...

其中坐标为 **归一化** 到 [0, 1] 的多边形顶点（x, y 交替）。

COCO instances 格式需要：bbox(x, y, w, h)、area、segmentation（均为绝对像素）。

用法::

    conda activate gbadmask
    cd /home/huachenghao/codes/GBADMask
    python datasets/prepare_coco_seg.py          # 默认处理 coco128-seg

输出目录结构::

    datasets/coco128-seg/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    └── val2017/
"""
import json
import os
import shutil
import sys
import zipfile

from PIL import Image

# COCO 80 类（Ultralytics 使用的 0-based 顺序）
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

HERE = os.path.dirname(os.path.abspath(__file__))
DL_DIR = os.path.join(HERE, "_dl")


def polygon_area(seg):
    """用 shoelace 公式计算多边形面积（COCO 的 area 字段语义）。"""
    xs = seg[0::2]
    ys = seg[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) / 2.0


def parse_yolo_seg_line(line, img_w, img_h):
    """解析一行 YOLO-seg 标注，返回 COCO annotation 的核心字段。"""
    parts = line.split()
    if len(parts) < 7:          # cls + 至少 3 个点(x,y) = 7
        return None
    cls_id = int(parts[0])
    coords = [float(v) for v in parts[1:]]
    # 保证是偶数个点（x, y 成对）
    if len(coords) % 2 != 0:
        coords = coords[:-1]
    seg = []
    for i in range(0, len(coords), 2):
        seg.append(coords[i] * img_w)
        seg.append(coords[i + 1] * img_h)
    xs = seg[0::2]
    ys = seg[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return {
        # COCO 的 category_id 从 1 开始（0 保留给背景）
        "category_id": cls_id + 1,
        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
        "area": polygon_area(seg),
        "segmentation": [seg],
        "iscrowd": 0,
    }


def convert_split(zip_path, out_root, src_split, out_split, cat_of_interest=None):
    """把压缩包里某个 split 的图片与标注转成 COCO json。

    Args:
        src_split: 压缩包内的 split 目录名（可能是 ``train`` 或 ``train2017``）
        out_split: 输出的 split 名，固定为 ``train`` / ``val``
    """
    # 注意：输出目录要拼在 out_split 后面（"train" + "2017"），
    # 不能用 src_split —— 后者在部分数据集里本身就叫 "train2017"，会拼成 "train20172017"
    img_out = os.path.join(out_root, "{}2017".format(out_split))
    os.makedirs(img_out, exist_ok=True)

    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    prefix = None
    for n in names:
        if "/images/" in n and n.endswith((".jpg", ".png")):
            prefix = n.split("/images/")[0]
            break
    if prefix is None:
        raise RuntimeError("压缩包中未找到 images 目录: {}".format(zip_path))

    img_names = sorted(
        n for n in names
        if n.startswith(prefix + "/images/") and n.endswith((".jpg", ".png"))
    )
    # 只取该 split 的图片（部分数据集按 train/val 分子目录）
    split_imgs = [n for n in img_names if "/{}/".format(src_split) in n]
    if not split_imgs:
        split_imgs = img_names

    images, annotations = [], []
    ann_id = 1
    for img_id, img_name in enumerate(split_imgs, start=1):
        base = os.path.basename(img_name)
        stem = os.path.splitext(base)[0]

        # 解压图片
        with zf.open(img_name) as src, open(os.path.join(img_out, base), "wb") as dst:
            shutil.copyfileobj(src, dst)

        with Image.open(os.path.join(img_out, base)) as im:
            w, h = im.size

        images.append({
            "id": img_id, "file_name": base, "width": w, "height": h,
        })

        # 对应的标注文件
        lbl_name = "{}/labels/{}/{}.txt".format(prefix, src_split, stem)
        if lbl_name not in names:
            lbl_alt = "{}/labels/{}.txt".format(prefix, stem)
            lbl_name = lbl_alt if lbl_alt in names else None
        if lbl_name is None:
            continue

        for line in zf.read(lbl_name).decode("utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            ann = parse_yolo_seg_line(line, w, h)
            if ann is None:
                continue
            if cat_of_interest is not None and ann["category_id"] not in cat_of_interest:
                continue
            ann.update({"id": ann_id, "image_id": img_id})
            annotations.append(ann)
            ann_id += 1

    return images, annotations


def build_categories(annotations):
    used = sorted({a["category_id"] for a in annotations})
    cats = []
    for cid in used:
        name = COCO80_NAMES[cid - 1] if 1 <= cid <= len(COCO80_NAMES) else "class_{}".format(cid)
        cats.append({"id": cid, "name": name, "supercategory": "object"})
    return cats


def prepare(dataset="coco128-seg", val_ratio=0.25):
    zip_path = os.path.join(DL_DIR, "{}.zip".format(dataset))
    if not os.path.isfile(zip_path):
        sys.exit("未找到 {}\n请先下载:\n  curl -L -o {} "
                 "https://ultralytics.com/assets/{}.zip".format(zip_path, zip_path, dataset))

    out_root = os.path.join(HERE, dataset)
    ann_root = os.path.join(out_root, "annotations")
    os.makedirs(ann_root, exist_ok=True)

    zf = zipfile.ZipFile(zip_path)
    has_split = any("/images/val/" in n for n in zf.namelist())

    if has_split:
        splits = {"train": "train", "val": "val"}
    else:
        splits = {"train": "train2017"}

    stats = {}
    for out_split, src_split in splits.items():
        images, annotations = convert_split(zip_path, out_root, src_split, out_split)
        cats = build_categories(annotations)
        json.dump(
            {"images": images, "annotations": annotations, "categories": cats},
            open(os.path.join(ann_root, "instances_{}2017.json".format(out_split)), "w"),
        )
        stats[out_split] = (len(images), len(annotations))
        print("  {}: {} 张图, {} 个实例".format(out_split, len(images), len(annotations)))

    # 无现成 val 划分时，从 train 中按比例切一部分作 val
    if not has_split and val_ratio > 0:
        tr_json = os.path.join(ann_root, "instances_train2017.json")
        va_json = os.path.join(ann_root, "instances_val2017.json")
        data = json.load(open(tr_json))
        imgs = data["images"]
        n_val = max(1, int(len(imgs) * val_ratio))
        val_ids = {im["id"] for im in imgs[-n_val:]}

        def subset(keep_val):
            sel = [im for im in imgs if (im["id"] in val_ids) == keep_val]
            ids = {im["id"] for im in sel}
            anns = [a for a in data["annotations"] if a["image_id"] in ids]
            return {"images": sel, "annotations": anns, "categories": data["categories"]}

        tr, va = subset(False), subset(True)
        json.dump(tr, open(tr_json, "w"))
        json.dump(va, open(va_json, "w"))
        # 把 val 图片复制到 val2017（与原图共存，磁盘开销可接受）
        val_dir = os.path.join(out_root, "val2017")
        os.makedirs(val_dir, exist_ok=True)
        for im in va["images"]:
            src = os.path.join(out_root, "train2017", im["file_name"])
            dst = os.path.join(val_dir, im["file_name"])
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        stats["val"] = (len(va["images"]), len(va["annotations"]))
        stats["train"] = (len(tr["images"]), len(tr["annotations"]))
        print("  已按 {:.0%} 划分 val -> train {} 张 / val {} 张"
              .format(val_ratio, stats["train"][0], stats["val"][0]))

    print("\n输出目录: {}".format(out_root))
    print("类别数: {}".format(len(build_categories(
        json.load(open(os.path.join(ann_root, "instances_train2017.json")))["annotations"]))))
    return out_root


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "coco128-seg"
    print("准备数据集: {}".format(ds))
    prepare(ds)
