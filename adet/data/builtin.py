import os

from detectron2.data import DatasetCatalog
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances

# datasets/ 目录所在的仓库根（adet/data/builtin.py -> 往上三级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_root(root):
    """相对路径按 CWD 解析，找不到时回退到仓库根，避免换目录启动就注册失败。"""
    if os.path.isabs(root) or os.path.isdir(root):
        return root
    return os.path.join(_PROJECT_ROOT, root)


# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "vintext_train": ("VinTextCustom/train_images", "VinTextCustom/train_data.json"),
    # "vintext_val": ("VinTextCustom/val_image", "VinTextCustom/val.json"),
    "vintext_val": ("VinTextCustom/val_image_rename", "VinTextCustom/val_rename.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),

}

metadata_text = {
    "thing_classes": ["text"]
}

# 小麦病害实例分割（HBueHxOW）
# 由 datasets/prepare_wheat_seg_coco.py 从 YOLO-seg 转出，目录结构见 README 第 6 节。
# 注册名 wheat_seg_{train,val,test}；train/val 也会被 tools/train_bl+.py 的
# 扫描逻辑注册（GBADMASK_DATA_ROOT 指向该目录时），两边都做了去重，不会冲突。
WHEAT_SEG_CLASSES = [
    "CrownAndRootRot",    # 根冠腐烂
    "HealthyWheat",       # 健康小麦
    "LeafRust",           # 叶锈病
    "PowderyMildew",      # 白粉病
    "WheatLooseSmut",     # 散黑穗病
    "WheatAphids",        # 蚜虫病
    "WheatCystNematode",  # 孢囊线虫病
    "WheatRedSpider",     # 红蜘蛛
    "WheatScab",          # 赤霉病
    "WheatSharpEyespot",  # 纹枯病
    "WheatStalkRot",      # 茎基腐
    "WheatTake-all",      # 全蚀病
]

_PREDEFINED_SPLITS_WHEAT = {
    "wheat_seg_train": (
        "HBueHxOW/wheat_seg/train2017",
        "HBueHxOW/wheat_seg/annotations/instances_train2017.json",
    ),
    "wheat_seg_val": (
        "HBueHxOW/wheat_seg/val2017",
        "HBueHxOW/wheat_seg/annotations/instances_val2017.json",
    ),
    "wheat_seg_test": (
        "HBueHxOW/wheat_seg/test2017",
        "HBueHxOW/wheat_seg/annotations/instances_test2017.json",
    ),
    # 按内容 MD5 去重 + 移除类别冲突后的版本（818 张，无验证集泄漏）
    # 由 prepare_wheat_seg_coco.py --dedup hash --drop-conflict 生成，见 DATA.md 第 7 节
    "wheat_seg_clean_train": (
        "HBueHxOW/wheat_seg_clean/train2017",
        "HBueHxOW/wheat_seg_clean/annotations/instances_train2017.json",
    ),
    "wheat_seg_clean_val": (
        "HBueHxOW/wheat_seg_clean/val2017",
        "HBueHxOW/wheat_seg_clean/annotations/instances_val2017.json",
    ),
    "wheat_seg_clean_test": (
        "HBueHxOW/wheat_seg_clean/test2017",
        "HBueHxOW/wheat_seg_clean/annotations/instances_test2017.json",
    ),
}

metadata_wheat = {
    # 注意：不要放 evaluator_type，register_coco_instances 内部已经固定传 "coco"
    "thing_classes": WHEAT_SEG_CLASSES,
}


def register_all_coco(root="datasets"):
    root = _resolve_root(root)
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    # datasets/ 未随仓库分发，数据缺失（或已由训练脚本注册过）时直接跳过
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_WHEAT.items():
        json_path = os.path.join(root, json_file)
        if key in DatasetCatalog.list() or not os.path.isfile(json_path):
            continue
        register_coco_instances(
            key,
            metadata_wheat,
            json_path,
            os.path.join(root, image_root),
        )


register_all_coco()