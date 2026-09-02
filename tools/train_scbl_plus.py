# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os, json, colorsys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets.coco import load_coco_json

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

# 不覆盖外部已设置的 CUDA_VISIBLE_DEVICES
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True

# 数据集路径
# 磁盘路径不再硬编码到某一台服务器：优先读环境变量 GBADMASK_DATA_ROOT，
# 未设置时回退到项目内的 datasets/Plantv2。目录结构见 README 第 6 节。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get(
    "GBADMASK_DATA_ROOT", os.path.join(_PROJECT_ROOT, "datasets", "Plantv2")
)
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train2017')   #train2017
VAL_PATH = os.path.join(DATASET_ROOT, 'val2017')    #val2017
TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train2017.json')       #
VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2017.json')          #


def _build_splits():
    """
    purpose: 扫描 GBADMASK_DATA_ROOT 及其同级目录，返回可用的 COCO instances 数据集。
             只做 os.path.isfile 检查，不读 json，因此数据未就绪时 import 不会失败。
    """
    splits = {}
    abs_root = os.path.abspath(DATASET_ROOT)
    for name in (os.path.basename(abs_root), "Strawberry"):
        root = os.path.join(os.path.dirname(abs_root), name)
        train_json = os.path.join(root, "annotations", "instances_train2017.json")
        val_json = os.path.join(root, "annotations", "instances_val2017.json")
        if os.path.isfile(train_json):
            splits[name + "_train"] = (os.path.join(root, "train2017"), train_json)
        if os.path.isfile(val_json):
            splits[name + "_val"] = (os.path.join(root, "val2017"), val_json)
    return splits


# 数据集的子集
# 注意：这里的数据集名字，需要更新到你的config文件中的 DATASETS 里.
PREDEFINED_SPLITS_DATASET = _build_splits()


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        print('key:', key, 'image_root:', image_root, 'json_file:', json_file)
        register_dataset_instances(name=key,
                                   metadate=get_dataset_instances_meta(json_file),
                                   json_file=json_file,
                                   image_root=image_root)


def get_dataset_instances_meta(json_file=TRAIN_JSON):
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    # 从 json 文件中读取类别（延迟加载，数据未就绪时不会在 import 阶段崩）
    with open(json_file, 'r') as f:
        DATASET_CATEGORIES = json.load(f)['categories']
    N = len(DATASET_CATEGORIES)  ###number classes , ignore background
    bright = True
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = (np.array(colors) * 255).astype('uint8')
    for i, category_info in enumerate(DATASET_CATEGORIES):
        category_info.update({"isthing": 1, "color": colors[i]})
        DATASET_CATEGORIES[i] = category_info

    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadate)


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """

    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.MODEL.FCPOSE_ON:
            mapper = FCPoseDatasetMapper(cfg, True)
        else:
            mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)
    register_dataset()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)  # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
