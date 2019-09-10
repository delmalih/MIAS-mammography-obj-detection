# Imports

import pandas as pd
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Functions

def read_annotations_file(path):
    pd_data = pd.read_table(path, delimiter=" ")
    pd_data = pd_data[pd_data.columns[:-1]]
    pd_data["path"] = pd_data["REFNUM"].map(lambda x: "%s.pgm" % x)
    return pd_data

def get_augmenter():
    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        )
    ])
    return seq

def data_augment(image, X, Y, R, factor):
    x1 = y1 = x2 = y2 = None
    if factor <= 1:
        if X==X and Y==Y and R==R:
            x1 = X - R
            y1 = Y - R
            x2 = X + R
            y2 = Y + R
        return image, [x1, y1, x2, y2]
    else:
        bboxes = []
        if X == X and Y == Y and R == R:
            bboxes.append(BoundingBox(x1=X, y1=Y, x2=X+R, y2=Y+R))
        bboxes = BoundingBoxesOnImage(bboxes, shape=image.shape)
        seq = get_augmenter()
        image_aug, bboxes_aug = seq(image=image, bounding_boxes=bboxes)
        for bbox in bboxes_aug.bounding_boxes:
            x1 = bbox.x1
            y1 = bbox.y1
            x2 = bbox.x2
            y2 = bbox.y2
        return image_aug, [x1, y1, x2, y2]