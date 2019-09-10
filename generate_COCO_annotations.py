# Imports

""" Global """
from glob import glob
import argparse
import os, shutil, json
from tqdm import tqdm
import cv2

""" Local """
from utils import read_annotations_file, data_augment

# Functions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", dest="images", help="Path to the images folder")
    parser.add_argument("-a", "--annotations", dest="annotations", help="Path to the .txt annotations file")
    parser.add_argument("-o", "--output", dest="output", help="Path to output COCO folder")
    parser.add_argument("--box", dest="box_class", help="Mix all classes to one 'box' class", action="store_true")
    parser.add_argument("--aug_fact", dest="augment_factor", help="Times for data augmentation", default=1)
    return parser.parse_args()

def create_required_folders(path):
    # if it exist already, reset the results directory
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.makedirs(path)

def generate_annotations(pd_data, images_folder, output_path, box_class=False, augment_factor=1):
    # Init. annotations
    annotations = {}
    annotations["categories"] = []
    annotations["images"] = []
    annotations["annotations"] = []

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

    # Images & Bboxes
    image_paths = glob(images_folder + "/*.pgm")
    for img_path in tqdm(image_paths):
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
            # Img writing
            img_path_jpg = "{}/{}{}.jpg".format(output_path, img_name, i)
            cv2.imwrite(img_path_jpg, images[i])
            
            # Img Annotations
            annotations["images"].append({
                "id": img_path.split("/")[-1][:-4],
                "width": 1024,
                "height": 1024,
                "file_name": img_path.split("/")[-1],
            })
            
            # Bbox Annotations
            if len(bbox):
                x1, y1, x2, y2 = bbox
                annotations["annotations"].append({
                    "id": "{}{}{}".format(img_name, i, CLASS_ID),
                    "category_id": CLASS_ID,
                    "image_id": img_name,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                })
    
    # Annotations writing
    with open("{}/annotations.json".format(output_path), "w") as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    args.augment_factor = int(args.augment_factor)
    create_required_folders(args.output)
    pd_data = read_annotations_file(args.annotations)
    generate_annotations(pd_data, args.images, args.output, box_class=args.box_class, augment_factor=args.augment_factor)