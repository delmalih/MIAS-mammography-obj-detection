# Imports

from glob import glob
import pandas as pd
import json
import argparse

# Functions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotations", dest="annotations", help="Path to the .txt file containing annotations")
    parser.add_argument("-o", "--output", dest="output", help="Path to output annotations file")
    return parser.parse_args()

def read_annotations_file(path):
    pd_data = pd.read_table(path, delimiter=" ")
    pd_data = pd_data[pd_data.columns[:-1]]
    pd_data["path"] = pd_data["REFNUM"].map(lambda x: "%s.pgm" % x)
    return pd_data

def generate_annotations(pd_data):
    annotations = {}

    # Images
    image_paths = pd_data["path"].values
    annotations["images"] = [{
        "id": img_path.split("/")[-1][:-4],
        "width": 1024,
        "height": 1024,
        "file_name": img_path.split("/")[-1],
    } for img_path in image_paths]

    # Categories
    classes = list(pd_data["CLASS"].value_counts().to_dict().keys())
    annotations["categories"] = [{
        "id": i,
        "name": c,
    } for i, c in enumerate(classes)]

    # Annotations
    annotations["annotations"] = []
    for img_path in image_paths:
        img_name = img_path.split("/")[-1][:-4]
        pd_img_data = pd_data[pd_data["REFNUM"] == img_name]
        X = pd_img_data["X"].values[0]
        Y = pd_img_data["Y"].values[0]
        RADIUS = pd_img_data["RADIUS"].values[0]
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
    
    return annotations

if __name__ == "__main__":
    args = parse_args()
    annotations_file_path = args.annotations
    pd_data = read_annotations_file(annotations_file_path)
    annotations = generate_annotations(pd_data)
    with open(args.output, "w") as f:
        json.dump(annotations, f, indent=4)