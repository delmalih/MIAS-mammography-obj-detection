# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
setup_logger()

# import some common libraries
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import json

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 inference")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file", required=True)
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--output", help="A folder to save output json file.", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score for instance predictions to be shown",)
    parser.add_argument("--weights_path", help="Path to the weights to load for inference.", required=True)
    return parser

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    # Set the model weights
    cfg.MODEL.WEIGHTS = args.weights_path

    cfg.freeze()

    return cfg

def do_inference(predictor, args):
    results = []
    for img_src in tqdm(args.input):
        img = cv2.imread(img_src)
        img_name = img_src.split("/")[-1]
        output = predictor(img)
        fields = output["instances"].get_fields()
        pred_boxes = fields["pred_boxes"]
        scores = fields["scores"]
        pred_classes = fields["pred_classes"]
        print(f"Image: {img_name} ! Found {len(pred_boxes)} instances !")
        for i, box in enumerate(pred_boxes):
            xmin, ymin, xmax, ymax = box
            results.append({
                "img_name": img_name,
                "bbox": [xmin, ymin, xmax-xmin, ymax-ymin],
                "score": scores[i],
                "category_id": pred_classes[i]
            })
    return results


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    results = do_inference(predictor, args)
    with open(f"{args.output}/results.json", "w") as outfile:
        json.dump(results, outfile, indent=4)