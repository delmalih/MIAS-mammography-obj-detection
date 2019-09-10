# Imports

from glob import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os, shutil
from tqdm import tqdm
import cv2

# Functions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", dest="images", help="Path to the images folder")
    parser.add_argument("-a", "--annotations", dest="annotations", help="Path to the .txt annotations file")
    parser.add_argument("-o", "--output", dest="output", help="Path to output VOC folder")
    parser.add_argument("--box", dest="box_class", help="Mix all classes to one 'box' class", action="store_true")
    return parser.parse_args()

def read_annotations_file(path):
    pd_data = pd.read_table(path, delimiter=" ")
    pd_data = pd_data[pd_data.columns[:-1]]
    pd_data["path"] = pd_data["REFNUM"].map(lambda x: "%s.pgm" % x)
    return pd_data

def create_required_folders(path):
    # if it exist already, reset the results directory
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.makedirs(path + "/JPEGImages")
    os.makedirs(path + "/Annotations")

def prettify(elem):
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generate_annotations(pd_data, images_folder, output_path, box_class=False):
    # Image conversion
    for img_path in tqdm(glob(images_folder + "/*.pgm"), desc="Image conversion"):
        img = cv2.imread(img_path)
        img_path_jpg = "{}/JPEGImages/{}.jpg".format(output_path, img_path.split("/")[-1])
        cv2.imwrite(img_path_jpg, img)

    # Annotation generation
    for img_path in tqdm(glob(images_folder + "/*.pgm"), desc="Annotation generation"):
        img_name = img_path.split("/")[-1][:-4]
        img_data = pd_data[pd_data["REFNUM"] == img_name]
        annotation = ET.Element("annotation")
        folder = ET.SubElement(annotation, "folder")
        folder.text = "JPEGImages"
        filename = ET.SubElement(annotation, "filename")
        filename.text = img_name + ".jpg"
        size = ET.SubElement(annotation, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = "1024"
        height_elem = ET.SubElement(size, "height")
        height_elem.text = "1024"
        depth_elem = ET.SubElement(size, "depth")
        depth_elem.text = "1"
        X = img_data["X"].values[0]
        Y = img_data["Y"].values[0]
        RADIUS = img_data["RADIUS"].values[0]
        CLASS = "box" if box_class else img_data["CLASS"].values[0]
        if X == X and Y == Y and RADIUS == RADIUS:
            obj = ET.SubElement(annotation, "object")
            name = ET.SubElement(obj, "name")
            name.text = CLASS
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(X))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(Y))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(X + RADIUS))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(Y + RADIUS))
        with open("{}/Annotations/{}.xml".format(output_path, img_name), "w") as xmlfile:
            xmlfile.write(prettify(annotation))

if __name__ == "__main__":
    args = parse_args()
    create_required_folders(args.output)
    pd_data = read_annotations_file(args.annotations)
    generate_annotations(pd_data, args.images, args.output, box_class=args.box_class)