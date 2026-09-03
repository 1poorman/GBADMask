# Copyright (c) GBADMask. All Rights Reserved.
"""
Copy-Paste 数据增广 —— 对应 ROADMAP M6.2 的 A1 组件。

实现思路（与官方 Copy-Paste / Simple Copy-Paste 一致）：
1. 启动时惰性构建一个 **donor 池**：从训练集随机采样 N 个实例，裁出每个实例
   的 (像素块, 掩膜, 相对坐标多边形, 相对 bbox, 类别)。
2. 每张训练图以概率 ``prob`` 随机抽取 ``k`` 个 donor，把它们**粘贴**到当前图
   的随机位置（用掩膜做合成，重叠区取 donor），并把对应实例标注追加进
   ``annotations``。
3. 粘贴发生在标准增广管线**之前**，因此粘贴进来的实例会一起参与后续的
   缩放/翻转/裁剪，与增广自然同步。

donor 池在进程内缓存（模块级 ``_CP_POOL``），dataloader 多 worker 时每个
worker 各自构建一份（首次调用时），之后复用，避免重复读盘。

注意：粘贴采用「mask 处覆盖」策略，可能让同一类别的多个实例重叠（这正是
Copy-Paste 提升小目标/稀少实例召回的关键），但这也使得粘贴实例互相重叠时
标注不再是两两不交的——对实例分割训练是可接受的（COCO 评估按实例计）。
"""
import logging
import random

import cv2
import numpy as np
import pycocotools.mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)

# 进程内 donor 池缓存： (dataset_name -> DonorPool)
_CP_POOL = {}


class DonorPool:
    """训练集实例裁剪池，按相对坐标存储以便粘贴到任意位置。"""

    def __init__(self, dataset_dicts, num_samples=2000, seed=0, min_size=8, min_fill=0.2):
        self.donors = []
        rng = random.Random(seed)
        idxs = list(range(len(dataset_dicts)))
        rng.shuffle(idxs)
        count = 0
        for i in idxs:
            if count >= num_samples:
                break
            d = dataset_dicts[i]
            try:
                img = utils.read_image(d["file_name"], format="BGR")
            except Exception as e:  # noqa: BLE001
                logger.warning("[CopyPaste] 跳过无法读取的图像 %s: %s", d.get("file_name"), e)
                continue
            h, w = img.shape[:2]
            for ann in d.get("annotations", []):
                if ann.get("iscrowd", 0) or "segmentation" not in ann:
                    continue
                box = BoxMode.convert(
                    ann["bbox"], ann["bbox_mode"], BoxMode.XYXY_ABS
                )
                bx0, by0, bx1, by1 = [int(round(float(v))) for v in box]
                bx0, by0 = max(0, bx0), max(0, by0)
                bx1, by1 = min(w, bx1), min(h, by1)
                cw, ch = bx1 - bx0, by1 - by0
                if bx1 <= bx0 or by1 <= by0 or min(cw, ch) < min_size:
                    continue
                segm = ann["segmentation"]
                if isinstance(segm, list):
                    rle = maskUtils.frPyObjects(segm, h, w)
                    rle = maskUtils.merge(rle)
                    full_mask = maskUtils.decode(rle).astype(np.uint8)
                else:
                    full_mask = maskUtils.decode(segm).astype(np.uint8)
                mask_crop = full_mask[by0:by1, bx0:bx1]
                if mask_crop.sum() < min_fill * cw * ch:
                    continue  # 裁剪框内多数是背景，不是干净的实例
                img_crop = img[by0:by1, bx0:bx1].copy()
                # 多边形转为相对坐标（相对裁剪框左上角），便于平移粘贴
                polys_rel = []
                if isinstance(segm, list):
                    for poly in segm:
                        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
                        pts[:, 0] -= bx0
                        pts[:, 1] -= by0
                        polys_rel.append(pts.flatten().tolist())
                else:
                    cnts, _ = cv2.findContours(
                        mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for c in cnts:
                        pts = c.reshape(-1, 2).astype(np.float64)
                        if pts.shape[0] >= 3:
                            polys_rel.append(pts.flatten().tolist())
                if not polys_rel:
                    continue
                self.donors.append(
                    {
                        "img": img_crop,
                        "mask": mask_crop,
                        "polys": polys_rel,
                        "cw": cw,
                        "ch": ch,
                        "cat_id": ann.get("category_id", 0),
                    }
                )
                count += 1
                if count >= num_samples:
                    break
        logger.info("[CopyPaste] donor 池构建完成，共 %d 个实例", len(self.donors))

    def sample(self, k, rng):
        if not self.donors:
            return []
        k = min(k, len(self.donors))
        return [self.donors[i] for i in rng.sample(range(len(self.donors)), k)]


def get_donor_pool(dataset_name, dataset_dicts, num_samples):
    if dataset_name not in _CP_POOL:
        _CP_POOL[dataset_name] = DonorPool(dataset_dicts, num_samples=num_samples)
    return _CP_POOL[dataset_name]


def apply_copy_paste(
    image,
    annotations,
    donor_pool,
    prob=0.5,
    max_donors=8,
    rng=None,
):
    """在原图上随机粘贴若干 donor 实例，返回 (新 image, 新 annotations)。

    新增的 annotation 使用绝对坐标（多边形已平移到粘贴位置），并自带
    ``bbox_mode = XYXY_ABS``，与现有标注走同一条 ``transform_instance_annotations``
    路径，因此后续增广会同步处理它们。

    Args:
        image: (H, W, 3) uint8 BGR
        annotations: 当前图的标注列表（dict 列表，原始格式）
        donor_pool: :class:`DonorPool`
        prob: 本次粘贴触发概率
        max_donors: 单次最多粘贴实例数
        rng: 随机数发生器（默认 ``random.Random()``）
    """
    if rng is None:
        rng = random.Random()
    if rng.random() >= prob or not donor_pool.donors:
        return image, annotations

    # detectron2 read_image 经 PIL 路径可能返回只读数组，粘贴前先复制
    if not image.flags.writeable:
        image = image.copy()

    h, w = image.shape[:2]
    k = rng.randint(1, max_donors)
    donors = donor_pool.sample(k, rng)
    new_anns = list(annotations)
    for d in donors:
        cw, ch = d["cw"], d["ch"]
        if cw > w or ch > h:
            continue
        dx = rng.randint(0, w - cw)
        dy = rng.randint(0, h - ch)
        m = d["mask"].astype(np.float32)[:, :, None]
        region = image[dy : dy + ch, dx : dx + cw]
        # mask 处用 donor 像素覆盖（重叠实例同样覆盖，符合 Copy-Paste 行为）
        image[dy : dy + ch, dx : dx + cw] = (
            (1.0 - m) * region + m * d["img"].astype(np.float32)
        ).astype(np.uint8)
        polys_abs = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in d["polys"]]
        polys_abs = [ (pts + np.array([dx, dy])).flatten().tolist() for pts in polys_abs ]
        new_anns.append(
            {
                "segmentation": polys_abs,
                "bbox": [dx, dy, dx + cw, dy + ch],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": d["cat_id"],
                "iscrowd": 0,
            }
        )
    return image, new_anns
