"""数据集注册：给一个**数据集名称**，即可注册出 ``<名称>_train/_val/_test``。

从 ``tools/train_bl+.py`` / ``tools/train_scbl_plus.py`` 抽出来的独立模块，训练脚本
只需::

    from register_datasets import register_from_cfg, apply_dataset

    apply_dataset(cfg, args.dataset)   # --dataset wheat_seg

目录约定见 ``base.py``：::

    <数据集目录>/
    ├── annotations/instances_{train,val,test}2017.json
    └── {train,val,test}2017/

数据集目录的查找顺序见 ``base.resolve_dataset_dir``：显式路径 > ``datasets/<名称>``
> ``GBADMASK_DATA_ROOT`` 同级目录 > 扫描 ``datasets/``。

数据集来源的优先级（见 :func:`apply_dataset`）：

1. 命令行 ``--dataset <名称>``
2. 配置文件 ``DATASETS.NAME: "<名称>"``
3. 从 ``DATASETS.TRAIN/TEST`` 的注册名反推（此时不改动 cfg）

命令行查看当前可用数据集::

    python tools/register_datasets.py            # 列出可用数据集
    python tools/register_datasets.py wheat_seg  # 注册并打印统计
"""
import json
import logging
import os
import sys

import colorsys
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base import (  # noqa: E402  需先补 sys.path 才能 import 同目录的 base
    DATASETS_DIR,
    SPLITS,
    ann_path,
    find_dataset_dirs,
    image_dir,
    resolve_dataset_dir,
    split_dataset_name,
    split_name,
)

# 数据集别名：名称 -> 相对 datasets/ 的目录。用于目录名与数据集名不一致的情况，
# 绝大多数数据集目录名即数据集名，靠 base.resolve_dataset_dir 自动找到即可。
DATASET_ALIASES = {
    "wheat_seg": "HBueHxOW/wheat_seg",
    "wheat_seg_clean": "HBueHxOW/wheat_seg_clean",
    # Plantv2 / Strawberry 已移至 datasets/ 下（深度 1），可直接被扫描发现，无需别名。
}

# 数据集类别数不一致时需要同步改的 config key
_NUM_CLASSES_KEYS = ("MODEL.FCOS.NUM_CLASSES", "MODEL.BASIS_MODULE.NUM_CLASSES")

logger = logging.getLogger(__name__)


def list_datasets():
    """当前 ``datasets/`` 下所有可用数据集名称（按字母序）。"""
    names = set(find_dataset_dirs())
    names.update(DATASET_ALIASES)
    return sorted(names)


def build_meta(json_file):
    """从 COCO json 的 ``categories`` 构造 detectron2 的 metadata。

    类别顺序沿用 json 中的顺序，并为每个类别生成可视化用的颜色。
    """
    with open(json_file, "r") as f:
        raw = json.load(f)["categories"]

    # 跳过语义背景类（如 "_background_"，category_id 通常为 0）：它不是可检测的 thing。
    # COCO / wheat 无此类别，跳过对它们无影响；Plantv2/Strawberry 有，必须剔除。
    categories = [c for c in raw if c.get("name", "").lower() != "_background_"]

    n = max(len(categories), 1)
    hsv = [(i / n, 1, 1.0) for i in range(n)]
    colors = (np.array([colorsys.hsv_to_rgb(*c) for c in hsv]) * 255).astype("uint8")

    thing_ids, thing_classes, thing_colors = [], [], []
    for i, cat in enumerate(categories):
        thing_ids.append(cat["id"])
        thing_classes.append(cat["name"])
        thing_colors.append(colors[i].tolist())

    return {
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "thing_dataset_id_to_contiguous_id": {k: i for i, k in enumerate(thing_ids)},
    }


def register_split(name, meta, json_file, image_root, verbose=True):
    """注册单个 split 到 DatasetCatalog / MetadataCatalog；已注册则跳过。"""
    if name in DatasetCatalog.list():
        return False
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        **meta,
    )
    if verbose:
        print("  注册 {} -> {}".format(name, json_file))
    return True


def register_dataset(name, verbose=True):
    """注册数据集 ``name`` 下所有存在的 split。

    Args:
        name: 数据集名称，如 ``wheat_seg``；也接受目录路径。
        verbose: 是否打印注册信息。

    Returns:
        dict: ``{split: 注册名}``，如 ``{"train": "wheat_seg_train", ...}``。
              各 split 缺失时不出现在结果里（并非所有数据集都有 test）。

    Raises:
        KeyError: 找不到该数据集（异常信息里会列出当前可用的数据集）。
    """
    dataset_dir = resolve_dataset_dir(DATASET_ALIASES.get(name, name))
    if dataset_dir is None:
        raise KeyError(
            "找不到数据集 '{}'。已查找 {} 及其子目录，当前可用：{}\n"
            "也可用绝对路径直接指定。".format(name, DATASETS_DIR, list_datasets() or "（无）")
        )

    # 类别元数据统一以 train 为准，保证 train/val/test 的 category_id 映射一致
    meta_json = next(
        (ann_path(dataset_dir, s) for s in SPLITS if os.path.isfile(ann_path(dataset_dir, s))),
        None,
    )
    meta = build_meta(meta_json)

    registered = {}
    for split in SPLITS:
        js = ann_path(dataset_dir, split)
        if not os.path.isfile(js):
            continue
        key = split_name(name, split)
        register_split(key, meta, js, image_dir(dataset_dir, split), verbose=verbose)
        registered[split] = key

    if verbose:
        print("数据集 {} ({}): {}".format(
            name, dataset_dir,
            ", ".join("{}->{}".format(s, k) for s, k in registered.items()) or "无可用 split"))
    return registered


def register_from_cfg(cfg, verbose=True):
    """按 config 里 ``DATASETS.TRAIN/TEST`` 的注册名反推数据集名并注册。

    例：``DATASETS.TRAIN = ("wheat_seg_train",)`` -> 注册 ``wheat_seg`` 的全部分 split。
    只注册，**不会**改动 cfg。
    """
    names = list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST)
    datasets = []
    for n in names:
        parsed = split_dataset_name(n)
        if parsed is None:
            continue
        dataset = DATASET_ALIASES.get(parsed[0], parsed[0])
        if dataset not in datasets:
            datasets.append(dataset)

    registered = {}
    for d in datasets:
        try:
            registered.update(register_dataset(d, verbose=verbose))
        except KeyError as e:
            # 数据集未就绪时给出明确提示，而不是等到 build_loader 阶段才报错
            raise KeyError("{}\nconfig 引用了数据集 {}，但磁盘上找不到。".format(e, d))
    return registered


def check_num_classes(cfg, registered):
    """数据集类别数与 ``MODEL.*.NUM_CLASSES`` 不一致时给出明确提示。

    **只提示不改配置**：静默改写会让「配置文件即事实」这条原则失效，
    排查问题时反而更难。类别数不一致通常表现为训练中途的 CUDA
    index 越界，错误信息完全看不出根因，所以在启动时就点明。
    """
    train = registered.get("train")
    if train is None:
        return
    n_actual = len(MetadataCatalog.get(train).thing_classes)
    for key in _NUM_CLASSES_KEYS:
        node = cfg
        for part in key.split("."):
            node = getattr(node, part, None)   # 非 BlendMask 系没有 BASIS_MODULE
            if node is None:
                break
        if node is not None and node != n_actual:
            logger.warning(
                "%s = %s，但数据集 %s 有 %d 个类别。请改为 %s: %d",
                key, node, train, n_actual, key, n_actual,
            )


def apply_dataset(cfg, name=None):
    """训练脚本的统一入口：注册数据集，并在需要时把 cfg.DATASETS 指过去。

    数据集来源优先级：命令行 ``--dataset`` > 配置 ``DATASETS.NAME`` >
    从 ``DATASETS.TRAIN/TEST`` 的注册名反推（此时不改动 cfg）。

    Args:
        cfg: 未 freeze 的 detectron2 config。
        name: 命令行 ``--dataset`` 给的数据集名称；为空时回退到 ``cfg.DATASETS.NAME``。

    Returns:
        dict: 已注册的 ``{split: 注册名}``。
    """
    # getattr 兜底：cfg 可能来自未升级的旧配置对象
    name = name or getattr(cfg.DATASETS, "NAME", "")
    if name:
        registered = register_dataset(name)
        # 显式指定了数据集（命令行或 yaml），覆盖 config 里的 DATASETS
        train = registered.get("train")
        if train:
            cfg.DATASETS.TRAIN = (train,)
        test = registered.get("val") or registered.get("test")
        if test:
            cfg.DATASETS.TEST = (test,)
        check_num_classes(cfg, registered)
        return registered

    registered = register_from_cfg(cfg)
    check_num_classes(cfg, registered)
    return registered


def _dataset_stats(name):
    """返回 [(split, 图片数, 实例数, 类别数)]，供命令行自检使用。"""
    dataset_dir = resolve_dataset_dir(DATASET_ALIASES.get(name, name))
    rows = []
    for split in SPLITS:
        js = ann_path(dataset_dir, split)
        if not os.path.isfile(js):
            continue
        with open(js) as f:
            data = json.load(f)
        rows.append((split, len(data["images"]), len(data["annotations"]),
                     len(data["categories"])))
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("{} 下可用的数据集:".format(DATASETS_DIR))
        found = list_datasets()
        if not found:
            print("  （无）先用 datasets/prepare_*.py 生成 COCO 格式数据")
            sys.exit(1)
        for n in found:
            print("  {:<16} {}".format(n, resolve_dataset_dir(DATASET_ALIASES.get(n, n))))
        sys.exit(0)

    for name in args:
        register_dataset(name)
        for split, n_img, n_ann, n_cat in _dataset_stats(name):
            print("  {:<5} {:>5} 张图  {:>5} 个实例  {} 类".format(
                split, n_img, n_ann, n_cat))
