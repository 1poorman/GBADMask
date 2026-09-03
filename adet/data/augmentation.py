import random

import cv2
import numpy as np
from fvcore.transforms import transform as T

from detectron2.data.transforms import Augmentation, RandomCrop, StandardAugInput, Transform
from detectron2.structures import BoxMode


class RandomScaleCrop(Augmentation):
    """Large Scale Jittering（LSJ）的轻量实现。

    本仓库依赖的 detectron2 版本未提供 ``RandomScale``，故这里用单个
    变换完成「随机缩放 + 固定尺寸随机裁剪」两步，等价于 LSJ 的核心行为：
    先以均匀采样的因子 ``s ~ U(scale_min, scale_max)`` 缩放整图（保持长宽比），
    再在缩放后的图上随机裁出 ``crop_size`` 大小的区域。

    对农作物病害这类**小尺寸**图像，原版 LSJ 的 [0.1, 2.0] 范围会过度破坏
    小目标，因此默认范围更保守（见 ``cfg.INPUT.LSJ``），由配置控制。

    该变换同时作用于 image / segmentation / coords，与 detectron2 的
    ``AugInput.apply_augmentations`` 管线兼容。
    """

    def __init__(self, scale_range=(0.3, 1.5), crop_size=(640, 640)):
        super().__init__()
        assert len(scale_range) == 2 and scale_range[0] <= scale_range[1]
        self.scale_range = tuple(scale_range)
        # crop_size: (h, w)
        self.crop_size = tuple(crop_size)

    def get_transform(self, img):
        return _ScaleCropTransform(self.scale_range, self.crop_size)


class _ScaleCropTransform(Transform):
    def __init__(self, scale_range, crop_size):
        super().__init__()
        self._h, self._w = crop_size  # (H, W)
        rng = np.random
        s = float(rng.uniform(*scale_range))
        self._s = s
        self._new_h, self._new_w = int(round(s * self._h)), int(round(s * self._w))
        # 缩放后可能比 crop 目标大（随机裁剪）或小（填充），两种情况分别处理
        if self._new_h >= self._h and self._new_w >= self._w:
            # 缩放后更大：在缩放图上随机裁出 crop 区域
            self._mode = "crop"
            self._x0 = int(rng.randint(0, self._new_w - self._w + 1))
            self._y0 = int(rng.randint(0, self._new_h - self._h + 1))
            self._px = self._py = 0
        else:
            # 缩放后更小：填充到 crop 尺寸，并把缩放图随机摆在画布内
            self._mode = "pad"
            self._px = int(rng.randint(0, max(0, self._w - self._new_w) + 1))
            self._py = int(rng.randint(0, max(0, self._h - self._new_h) + 1))
            self._x0 = self._y0 = 0

    def _place(self, resized, is_seg):
        C = resized.shape[2] if resized.ndim == 3 else 1
        if self._mode == "crop":
            return resized[self._y0 : self._y0 + self._h, self._x0 : self._x0 + self._w]
        # pad：把缩放图贴到零画布的随机位置
        canvas = np.zeros((self._h, self._w, C) if C > 1 else (self._h, self._w),
                          dtype=resized.dtype)
        canvas[self._py : self._py + self._new_h, self._px : self._px + self._new_w] = resized
        return canvas

    def apply_image(self, img):
        if img.ndim == 3:
            resized = cv2.resize(
                img, (self._new_w, self._new_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            resized = cv2.resize(
                img, (self._new_w, self._new_h), interpolation=cv2.INTER_NEAREST
            )
        return self._place(resized, is_seg=False)

    def apply_segmentation(self, seg):
        resized = cv2.resize(
            seg.astype(np.uint8),
            (self._new_w, self._new_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return self._place(resized, is_seg=True)

    def apply_coords(self, coords):
        coords = coords * self._s
        if self._mode == "crop":
            coords[:, 0] -= self._x0
            coords[:, 1] -= self._y0
        else:
            coords[:, 0] += self._px
            coords[:, 1] += self._py
        return coords


def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                raise ValueError(
                    "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                        len(instances)
                    )
                )
                return T.CropTransform(0, 0, image_size[1], image_size[0])

    return T.CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    """ Instance-aware cropping.
    """

    def __init__(self, crop_type, crop_size, crop_instance=True):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance
        self.input_args = ("image", "boxes")

    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )
