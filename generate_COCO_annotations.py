# Imports

from glob import glob
import pandas as pd
import argparse
import os, shutil, json
from tqdm import tqdm
import cv2

# Functions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", dest="images", help="Path to the images folder")
    parser.add_argument("-a", "--annotations", dest="annotations", help="Path to the .txt annotations file")
    parser.add_argument("-o", "--output", dest="output", help="Path to output COCO folder")
    parser.add_argument("--box", dest="box_class", help="Mix all classes to one 'box' class", action="store_true")
    return parser.parse_args()

def create_required_folders(path):
    # if it exist already, reset the results directory
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.makedirs(path)

def read_annotations_file(path):
    pd_data = pd.read_table(path, delimiter=" ")
    pd_data = pd_data[pd_data.columns[:-1]]
    pd_data["path"] = pd_data["REFNUM"].map(lambda x: "%s.pgm" % x)
    return pd_data

def generate_annotations(pd_data, images_folder, output_path, box_class=False):
    # Init. annotations
    annotations = {}

    # Images
    image_paths = glob(images_folder + "/*.pgm")
    annotations["images"] = []
    for img_path in tqdm(image_paths, desc="Image conversion"):
        img = cv2.imread(img_path)
        img_path_jpg = "{}/{}.jpg".format(output_path, img_path.split("/")[-1])
        cv2.imwrite(img_path_jpg, img)
        annotations["images"].append({
            "id": img_path.split("/")[-1][:-4],
            "width": 1024,
            "height": 1024,
            "file_name": img_path.split("/")[-1],
        })

    # Categories
    if box_class:
        classes = ["box"]
        annotations["categories"] = [{
            "id": 0,
            "name": "box",
        }]
    else:
        classes = list(pd_data["CLASS"].value_counts().to_dict().keys())
        annotations["categories"] = [{
            "id": i,
            "name": c,
        } for i, c in enumerate(classes)]

    # Annotations
    annotations["annotations"] = []
    for img_path in tqdm(image_paths, desc="Annotation generation"):
        img_name = img_path.split("/")[-1][:-4]
        pd_img_data = pd_data[pd_data["REFNUM"] == img_name]
        X = pd_img_data["X"].values[0]
        Y = pd_img_data["Y"].values[0]
        RADIUS = pd_img_data["RADIUS"].values[0]
        if box_class:
            CLASS_ID = 0
        else:
            CLASS = pd_img_data["CLASS"].values[0]
            CLASS_ID = classes.index(CLASS)
        if X == X and Y == Y and RADIUS == RADIUS:
            annotations["annotations"].append({
                "id": "{}{}".format(img_name, CLASS_ID),
                "category_id": CLASS_ID,
                "image_id": img_name,
                "area": RADIUS**2,
                "bbox": [X - RADIUS, Y - RADIUS, RADIUS, RADIUS],
            })
    
    with open("{}/annotations.json".format(output_path), "w") as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    create_required_folders(args.output)
    pd_data = read_annotations_file(args.annotations)
    generate_annotations(pd_data, args.images, args.output, box_class=args.box_class)