# Imports

""" Global """
from glob import glob
import argparse
import os, shutil, json
from tqdm import tqdm
import cv2
import numpy as np

""" Local """
from utils import read_annotations_file, data_augment

# Functions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", dest="images", help="Path to the images folder", required=True)
    parser.add_argument("-a", "--annotations", dest="annotations", help="Path to the .txt annotations file", required=True)
    parser.add_argument("-o", "--output", dest="output", help="Path to output COCO folder", required=True)
    parser.add_argument("--box", dest="box_class", help="Mix all classes to one 'box' class", action="store_true")
    parser.add_argument("--aug_fact", dest="augment_factor", help="Times for data augmentation", default=1)
    parser.add_argument("--train_val_split", dest="train_val_split", help="Percetange of the train folder (default 0.9)", default=0.9)
    return parser.parse_args()

def create_required_folders(path):
    # if it exist already, reset the results directory
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.makedirs(path + "/images/train")
    os.makedirs(path + "/images/val")
    os.makedirs(path + "/annotations")

def generate_annotations(pd_data, images_folder, output_path, box_class=False, augment_factor=1, train_val_split=0.9):
    # Init. train annotations
    train_annotations = {}
    train_annotations["images"] = []
    train_annotations["annotations"] = []

    # Init. val annotations
    val_annotations = {}
    val_annotations["images"] = []
    val_annotations["annotations"] = []

    # Categories
    if box_class:
        classes = ["box"]
        class_annotations = [{
            "id": 0,
            "name": "box",
        }]
    else:
        classes = list(pd_data["CLASS"].value_counts().to_dict().keys())
        class_annotations = [{
            "id": i,
            "name": c,
        } for i, c in enumerate(classes)]
    train_annotations["categories"] = class_annotations
    val_annotations["categories"] = class_annotations

    # Images & Bboxes
    image_paths = glob(images_folder + "/*.pgm")
    for n, img_path in enumerate(tqdm(image_paths)):
        # Getting data
        img = cv2.imread(img_path)
        img_name = img_path.split("/")[-1][:-4]
        img_data = pd_data[pd_data["REFNUM"] == img_name]
        X = img_data["X"].values[0]
        Y = img_data["Y"].values[0]
        R = img_data["RADIUS"].values[0]
        if box_class:
            CLASS_ID = 0
        else:
            CLASS = img_data["CLASS"].values[0]
            CLASS_ID = classes.index(CLASS)
        
        # Augmentation
        images, bboxes = data_augment(img, X, Y, R, augment_factor)
        
        for i, bbox in enumerate(bboxes):
            # Image data
            img_id = int(float("{}{}".format(n, i)))
            final_img_name = "{}{}.jpg".format(img_name, i)
            is_train = np.random.random() < train_val_split

            # Img Annotations
            image_annotation = {
                "id": img_id,
                "width": 1024,
                "height": 1024,
                "file_name": final_img_name,
            }
            if is_train:
                img_path_jpg = "{}/images/train/{}".format(output_path, final_img_name)
                train_annotations["images"].append(image_annotation)
            else:
                img_path_jpg = "{}/images/val/{}".format(output_path, final_img_name)
                val_annotations["images"].append(image_annotation)
            cv2.imwrite(img_path_jpg, images[i])
            
            # Bbox Annotations
            if len(bbox):
                x1, y1, x2, y2 = map(int, bbox)
                assert x1 >= 0 and y1 >= 0 and x2 < 1024 and y2 < 1024
                bbox_annotation = {
                    "id": "{}{}{}".format(img_id, CLASS_ID, i),
                    "category_id": CLASS_ID,
                    "image_id": img_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "iscrowd": False,
                    "area": (x2 - x1) * (y2 - y1),
                    "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]],
                }
                if is_train:
                    train_annotations["annotations"].append(bbox_annotation)
                else:
                    val_annotations["annotations"].append(bbox_annotation)
    
    # Annotations writing
    with open("{}/annotations/instances_train.json".format(output_path), "w") as f:
        json.dump(train_annotations, f, indent=4)
    with open("{}/annotations/instances_val.json".format(output_path), "w") as f:
        json.dump(val_annotations, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    args.augment_factor = int(args.augment_factor)
    args.train_val_split = float(args.train_val_split)
    create_required_folders(args.output)
    pd_data = read_annotations_file(args.annotations)
    generate_annotations(pd_data, args.images, args.output, box_class=args.box_class,
                         augment_factor=args.augment_factor, train_val_split=args.train_val_split)