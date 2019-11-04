import os
import torch
import argparse

def remove_key(dict, key):
    if key in dict:
        dict.pop(key)

parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    help="path to save the converted model",
    type=str,
)

args = parser.parse_args()
model = torch.load(args.pretrained_path)

remove_key(model, 'iteration')
remove_key(model, 'scheduler')
remove_key(model, 'optimizer')

remove_key(model['model'], 'module.roi_heads.box.predictor.cls_score.weight')

remove_key(model['model'], 'module.roi_heads.box.predictor.cls_score.weight')
remove_key(model['model'], 'module.roi_heads.box.predictor.cls_score.bias')
remove_key(model['model'], 'module.roi_heads.box.predictor.bbox_pred.weight')
remove_key(model['model'], 'module.roi_heads.box.predictor.bbox_pred.bias')

remove_key(model['model'], 'module.roi_heads.mask.predictor.mask_fcn_logits.weight')
remove_key(model['model'], 'module.roi_heads.mask.predictor.mask_fcn_logits.bias')

remove_key(model['model'], 'module.rpn.head.cls_logits.weight')
remove_key(model['model'], 'module.rpn.head.cls_logits.bias')
remove_key(model['model'], 'module.rpn.head.bbox_pred.weight')
remove_key(model['model'], 'module.rpn.head.bbox_pred.bias')

remove_key(model['model'], 'cls_score.bias')
remove_key(model['model'], 'cls_score.weight')
remove_key(model['model'], 'bbox_pred.bias')
remove_key(model['model'], 'bbox_pred.weight')

torch.save(model, args.save_path)
print('saved to {}.'.format(args.save_path))