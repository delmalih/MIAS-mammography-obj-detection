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
import os
from collections import OrderedDict
import torch
import json

from detectron2.structures import BoxMode
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


COCO_PATH = "/home/ubuntu/mias-db/COCO"

def get_coco_dicts(coco_dir, train_val):
    json_file = os.path.join(coco_dir, "annotations", "instances_{}.json".format(train_val))
    with open(json_file) as f:
        ANNOTATIONS = json.load(f)

    def convert_bbox(bbox):
        x, y, w, h = bbox
        return x - w, y - h, x + w, y + h

    dataset_dicts = []
    for img_data in ANNOTATIONS["images"]:
        record = {}
        record["file_name"] = os.path.join(coco_dir, "images", train_val, img_data["file_name"])
        record["height"] = img_data["height"]
        record["width"] = img_data["width"]
        record["annotations"] = [{
            "bbox": convert_bbox(box["bbox"]),
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": box["segmentation"],
            "category_id": box["category_id"],
            "iscrowd": 0,
        } for box in filter(lambda x: x["image_id"] == img_data["id"], ANNOTATIONS["annotations"])]
        dataset_dicts.append(record)
    
    return dataset_dicts

def get_categories(coco_dir, train_val):
    json_file = os.path.join(coco_dir, "annotations", "instances_{}.json".format(train_val))
    with open(json_file) as f:
        ANNOTATIONS = json.load(f)
    
    categories = ANNOTATIONS["categories"]
    return [cat["name"] for cat in categories]

def register_dataset():
    for d in ["train", "val"]:
        thing_classes = get_categories(COCO_PATH, d)
        dataset_name = "{}_{}".format("mias", d)
        DatasetCatalog.register(dataset_name, lambda d=d: get_coco_dicts(COCO_PATH, d))
        MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

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
        if evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
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
        logger = logging.getLogger("detectron2.trainer")
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
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
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
    register_dataset()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


#     cfg.DATASETS.TRAIN = ("{}_train".format(args.dataset_name),)
#     cfg.DATASETS.TEST = ("{}_val".format(args.dataset_name),)
#     # cfg.DATASETS.TEST = ()
#     cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.MODEL.WEIGHTS = base_cfg_to_model_path[args.base_cfg]
#     cfg.SOLVER.IMS_PER_BATCH = args.images_per_batch
#     cfg.SOLVER.BASE_LR = 0.00025
#     cfg.SOLVER.MAX_ITER = args.max_iter
#     cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(get_categories(args.coco_path, "train"))
#     cfg.OUTPUT_DIR = args.output_dir
#     cfg.RESUME_TRAINING = args.resume_training