import copy
import logging
import os.path as osp

import cv2
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .augmentation import RandomCropWithInstance
from .detection_utils import (annotations_to_instances, build_augmentation,
                              transform_instance_annotations)

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        # cfg.INPUT.CROP.ENABLED = False
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        # "npz"  : 只从预生成的 thing_train2017/*.npz 读取（官方行为，自定义数据集没有）
        # "auto" : 优先 npz，不存在时从当前图的 gt_masks 在线合成
        # "online": 始终在线合成，完全不依赖 npz
        self.basis_sem_source = cfg.MODEL.BASIS_MODULE.SEM_SOURCE
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            basis_sem_gt = self._load_basis_sem(dataset_dict)
            if basis_sem_gt is None:
                # 从当前图的实例掩膜在线合成语义标签
                basis_sem_gt = self._make_basis_sem_from_instances(dataset_dict)
            if basis_sem_gt is not None:
                dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

    def _load_basis_sem(self, dataset_dict):
        """从预生成的 npz 读取 basis 语义标签，不存在时返回 None。"""
        if self.basis_sem_source == "online":
            return None
        if self.ann_set == "coco":
            basis_sem_path = (
                dataset_dict["file_name"]
                .replace("train2017", "thing_train2017")
                .replace("image/train", "thing_train")
            )
        else:
            basis_sem_path = (
                dataset_dict["file_name"]
                .replace("coco", "lvis")
                .replace("train2017", "thing_train")
            )
        basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
        if not osp.isfile(basis_sem_path):
            return None
        basis_sem_gt = np.load(basis_sem_path)["mask"]
        basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
        return torch.as_tensor(basis_sem_gt.astype("long"))

    @staticmethod
    def _make_basis_sem_from_instances(dataset_dict):
        """从 gt_masks 在线合成 basis 语义标签图。

        官方实现要求预先把 COCO 的 instance mask 烘成 ``thing_train2017/*.npz``，
        自定义数据集（小麦病害等）没有这一步，导致 ``BASIS_MODULE.LOSS_ON`` 只能关
        掉，basis module 的语义辅助损失（本项目为 Focal-Dice-CE）完全不参与训练。

        这里直接从 dataloader 已有的 ``instances.gt_masks`` 合成：把每个实例掩膜
        覆盖的像素标为 ``类别 id + 1``（0 保留给背景），与 COCO 语义标签的约定一致。
        标注已过 transform，因此天然与增广同步，也无需额外磁盘空间。
        """
        instances = dataset_dict.get("instances")
        if instances is None or not instances.has("gt_masks"):
            return None
        if len(instances) == 0:
            return None

        masks = instances.gt_masks
        # BitMasks -> (N, H, W) bool；PolygonMasks 需先转成密集掩膜
        from detectron2.structures import PolygonMasks
        if hasattr(masks, "tensor"):                      # BitMasks
            m = masks.tensor
        elif isinstance(masks, PolygonMasks):
            polys = masks.polygons          # N 个实例，每个是若干段 polygon
            h, w = dataset_dict["height"], dataset_dict["width"]
            dense = np.zeros((len(polys), h, w), dtype=np.uint8)
            for i, parts in enumerate(polys):
                for seg in parts:
                    pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
                    if pts.shape[0] < 3:
                        continue
                    # 顶点需在图内，否则 fillPoly 会静默越界
                    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                    cv2.fillPoly(dense[i], [pts.astype(np.int32)], 1)
            m = torch.as_tensor(dense, dtype=torch.bool)
        else:
            m = torch.as_tensor(np.asarray(masks), dtype=torch.bool)
        if m.numel() == 0:
            return None

        h, w = m.shape[-2:]
        sem = torch.zeros((h, w), dtype=torch.long, device=m.device)
        classes = instances.gt_classes.to(m.device)
        for i in range(m.shape[0]):
            # 后写的覆盖先写的：COCO 语义分割标签的惯例（小物体后画）
            sem[m[i]] = classes[i].long() + 1
        return sem
