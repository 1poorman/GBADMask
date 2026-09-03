import random

import cv2
import numpy as np
from fvcore.transforms import transform as T

from detectron2.data.transforms import Augmentation, RandomCrop, StandardAugInput, Transform
from detectron2.structures import BoxMode


class RandomScaleCrop(Augmentation):
    """Large Scale Jittering（LSJ）实现。

    行为（与标准 LSJ 一致）：
    1. 以 ``s ~ U(scale_min, scale_max)`` **等比**缩放整图（保持纵横比）；
    2. 把缩放后的图贴到 ``max(缩放尺寸, crop_size)`` 的零画布随机位置；
    3. 从画布上随机裁出 ``crop_size`` 大小的区域。

    缩放后比 crop 目标大 → 等价随机裁剪；比目标小 → 等价随机填充；
    横纵分别超出/不足（非方形图常见）也能正确处理。

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
        return _ScaleCropTransform(self.scale_range, self.crop_size, img.shape[:2])


class _ScaleCropTransform(Transform):
    def __init__(self, scale_range, crop_size, orig_hw):
        super().__init__()
        self._H, self._W = crop_size      # 输出尺寸 (h, w)
        rng = np.random
        s = float(rng.uniform(*scale_range))
        self._s = s
        # 等比缩放后的整图尺寸（至少 1 像素）
        self._new_h = max(1, int(round(s * orig_hw[0])))
        self._new_w = max(1, int(round(s * orig_hw[1])))
        # 画布 = max(缩放图, crop)，缩放图随机摆放，再从中随机裁 crop 区域
        self._ch = max(self._H, self._new_h)
        self._cw = max(self._W, self._new_w)
        self._px = int(rng.randint(0, self._cw - self._new_w + 1))
        self._py = int(rng.randint(0, self._ch - self._new_h + 1))
        self._x0 = int(rng.randint(0, self._cw - self._W + 1))
        self._y0 = int(rng.randint(0, self._ch - self._H + 1))

    def _resize(self, img, interp):
        return cv2.resize(img, (self._new_w, self._new_h), interpolation=interp)

    def _place_and_crop(self, resized):
        if resized.ndim == 3:
            canvas = np.zeros((self._ch, self._cw, resized.shape[2]),
                              dtype=resized.dtype)
        else:
            canvas = np.zeros((self._ch, self._cw), dtype=resized.dtype)
        canvas[self._py : self._py + self._new_h,
               self._px : self._px + self._new_w] = resized
        return canvas[self._y0 : self._y0 + self._H,
                      self._x0 : self._x0 + self._W]

    def apply_image(self, img):
        resized = self._resize(img, cv2.INTER_LINEAR if img.ndim == 3
                               else cv2.INTER_NEAREST)
        return self._place_and_crop(resized)

    def apply_segmentation(self, seg):
        resized = self._resize(seg.astype(np.uint8), cv2.INTER_NEAREST)
        return self._place_and_crop(resized)

    def apply_coords(self, coords):
        coords = coords.astype(np.float64)
        coords[:, 0] = coords[:, 0] * self._s + self._px - self._x0
        coords[:, 1] = coords[:, 1] * self._s + self._py - self._y0
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
