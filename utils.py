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
        iaa.SomeOf((1, 4), [
            iaa.CropAndPad(percent=(-0.25, 0.25)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.Add((-20, 20)),
            iaa.AddElementwise((-10, 10)),
            iaa.AdditiveGaussianNoise(scale=0.01*255),
            iaa.Multiply((0.8, 1.2)),
            iaa.Affine(shear=(-8, 8)),
            iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)}),
            iaa.geometric.Rot90((0, 4)),
        ])
    ])
    return seq

def data_augment(image, X, Y, R, factor):
    images = []
    bboxes = []
    x1 = y1 = x2 = y2 = None
    if factor <= 1:
        if X==X and Y==Y and R==R:
            x1 = X - R
            y1 = 1024 - (Y - R)
            x2 = X + R
            y2 = 1024 - (Y + R)
            bboxes.append([x1, y1, x2, y2])
        else:
            bboxes.append([])
        images.append(image)
    else:
        seq = get_augmenter()
        bbs_before_aug = []
        if X == X and Y == Y and R == R:
            bbs_before_aug.append(BoundingBox(x1=X, y1=Y, x2=X+R, y2=Y+R))
        bbs_before_aug = BoundingBoxesOnImage(bbs_before_aug, shape=image.shape)
        for _ in range(factor):
            image_aug, bbs_after_aug = seq(image=image, bounding_boxes=bbs_before_aug)
            images.append(image_aug)
            if len(bbs_after_aug.bounding_boxes):
                x1 = bbs_after_aug.bounding_boxes[0].x1
                y1 = bbs_after_aug.bounding_boxes[0].y1
                x2 = bbs_after_aug.bounding_boxes[0].x2
                y2 = bbs_after_aug.bounding_boxes[0].y2
                bboxes.append([x1, y1, x2, y2])
            else:
                bboxes.append([])
    return images, bboxes