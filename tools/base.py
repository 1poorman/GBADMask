"""GBADMask 脚本的公共基础配置：路径、环境变量与 COCO 目录命名约定。

把原先散落在 ``tools/train_bl+.py`` / ``tools/train_scbl_plus.py`` 里的路径硬编码
集中到一处，训练脚本只通过**数据集名称**取数据，不再关心磁盘布局。

三类内容：
  1. 路径常量（项目根、配置、数据集、输出、预训练权重）
  2. 环境变量初始化（``CUDA_VISIBLE_DEVICES`` / ``KMP_DUPLICATE_LIB_OK`` / cuDNN）
  3. COCO 目录命名约定 + 名称到目录的解析

用法::

    from base import PROJECT_ROOT, DATASETS_DIR, setup_env, resolve_dataset_dir

环境变量：
    ``GBADMASK_DATA_ROOT``  当前使用的数据集根目录，优先级高于本模块的默认推导
    ``CUDA_VISIBLE_DEVICES`` 可见 GPU；本模块用 ``setdefault``，**不覆盖**外部设置
"""
import os

import torch

# --------------------------------------------------------------------------
# 路径（tools/base.py -> 往上一级为项目根）
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
PRETRAIN_DIR = os.path.join(PROJECT_ROOT, "pretrain")

_ENV_DATA_ROOT = "GBADMASK_DATA_ROOT"

# 未通过 --dataset 指定、且 config 里也读不出数据集名时的兜底
DEFAULT_DATASET = "wheat_seg"

# --------------------------------------------------------------------------
# COCO 目录约定（与 README 第 6 节一致）
#   <数据集根>/
#     ├── annotations/instances_{train,val,test}2017.json
#     └── {train,val,test}2017/
# --------------------------------------------------------------------------
ANN_DIR_NAME = "annotations"

# 顺序即优先级：train 最先，类别元数据以 train 为准；--dataset 注册的 TEST 也优先取 val
SPLITS = ("train", "val", "test")

# 扫描 datasets/ 查找数据集目录时的最大深度（datasets/HBueHxOW/wheat_seg 为 2 层）
_MAX_SCAN_DEPTH = 3
# 扫描时跳过的目录（下载缓存等）
_SCAN_SKIP = {"_dl", "__pycache__", ".git", ".ipynb_checkpoints"}


def json_name(split):
    """``train`` -> ``instances_train2017.json``"""
    return "instances_{}2017.json".format(split)


def image_dir_name(split):
    """``train`` -> ``train2017``"""
    return "{}2017".format(split)


def ann_path(dataset_dir, split):
    """数据集目录下某个 split 的标注 json 路径。"""
    return os.path.join(dataset_dir, ANN_DIR_NAME, json_name(split))


def image_dir(dataset_dir, split):
    """数据集目录下某个 split 的图片目录路径。"""
    return os.path.join(dataset_dir, image_dir_name(split))


def split_name(dataset, split):
    """``("wheat_seg", "train")`` -> ``wheat_seg_train``（detectron2 的注册名）"""
    return "{}_{}".format(dataset, split)


def split_dataset_name(registered_name):
    """``wheat_seg_train`` -> ``("wheat_seg", "train")``；不含已知 split 则返回 None。

    用于从 config 的 ``DATASETS.TRAIN`` 反推数据集名称。
    """
    for s in SPLITS:
        suffix = "_" + s
        if registered_name.endswith(suffix) and len(registered_name) > len(suffix):
            return registered_name[: -len(suffix)], s
    return None


# --------------------------------------------------------------------------
# 环境
# --------------------------------------------------------------------------
def setup_env(devices="0,1,2"):
    """初始化运行时环境变量与 cuDNN。

    Args:
        devices: ``CUDA_VISIBLE_DEVICES`` 的默认值。用 ``setdefault``，
                 外部（命令行 / 调度系统）已经设过时不覆盖。
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", devices)
    # OMP 与 libiomp 冲突时会让进程直接 abort，必须允许重复加载
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 输入尺寸固定（detectron2 会按 MIN/MAX_SIZE 缩放），开 benchmark 更快
    torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------
# 名称 -> 目录
# --------------------------------------------------------------------------
def _is_dataset_dir(path):
    """目录下存在任一 split 的 COCO instances json 即认为是数据集目录。"""
    if not os.path.isdir(path):
        return False
    return any(os.path.isfile(ann_path(path, s)) for s in SPLITS)


def find_dataset_dirs(root=DATASETS_DIR, max_depth=_MAX_SCAN_DEPTH):
    """扫描 ``root``，返回 ``{数据集名称: 目录}``。

    只检查目录名与 json 是否存在，不读 json 内容，因此在数据未就绪或目录很大时
    调用开销也可接受。同名目录取**层级更浅**的那个。
    """
    found = {}
    if not os.path.isdir(root):
        return found

    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SCAN_SKIP)
        depth = dirpath.count(os.sep) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        if _is_dataset_dir(dirpath):
            name = os.path.basename(dirpath)
            # 已存在说明之前找到的层级更浅（os.walk 自顶向下），保留之
            found.setdefault(name, dirpath)
    return found


def get_data_root(name=None):
    """当前数据集根目录。

    优先级：``GBADMASK_DATA_ROOT`` 环境变量 > ``datasets/<name>`` >
    ``datasets/<DEFAULT_DATASET>``。
    """
    env = os.environ.get(_ENV_DATA_ROOT)
    if env:
        return os.path.abspath(env)
    if name:
        return os.path.join(DATASETS_DIR, name)
    return os.path.join(DATASETS_DIR, DEFAULT_DATASET)


def resolve_dataset_dir(name, search=True):
    """数据集名称 -> 目录；找不到返回 ``None``。

    依次尝试：绝对路径 / 相对当前工作目录的路径 / ``datasets/<name>`` /
    ``GBADMASK_DATA_ROOT`` 的父目录下的 ``<name>`` / 扫描 ``datasets/``。
    """
    if not name:
        return None
    # 1) 显式路径（绝对或相对 CWD）
    path = os.path.abspath(os.path.expanduser(name))
    if _is_dataset_dir(path):
        return path
    # 2) datasets/<name>
    path = os.path.join(DATASETS_DIR, name)
    if _is_dataset_dir(path):
        return path
    if not search:
        return None
    # 3) GBADMASK_DATA_ROOT 的同级目录（旧版 train_*.py 的扫描行为）
    env = os.environ.get(_ENV_DATA_ROOT)
    if env:
        path = os.path.join(os.path.dirname(os.path.abspath(env)), name)
        if _is_dataset_dir(path):
            return path
    # 4) 扫描 datasets/（支持 datasets/HBueHxOW/wheat_seg 这类嵌套布局）
    return find_dataset_dirs().get(name)


def ensure_dir(path):
    """建目录（若不存在），并返回该路径。"""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path
