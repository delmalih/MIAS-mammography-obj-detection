# Imports

from maskrcnn_benchmark.structures.bounding_box import BoxList
import imageio
import numpy as np
from PIL import Image
import torch
import argparse
import json

# Dataset

class MIASDataset(object):
    def __init__(self, coco_dataset_path, train_or_val="train"):
        annotations_path = "{}/annotations/instances_{}.json".format(coco_dataset_path, train_or_val)
        with open(annotations_path, "r") as json_annotations:
            coco_annotations = json.load(json_annotations)
        
        self.height = 1024
        self.width = 1024
        self.images_path = coco_dataset_path
        self.images = coco_annotations["images"]
        self.classes = coco_annotations["categories"]
        self.bounding_boxes = {}

        for bbox in coco_annotations["annotations"]:
            image_name = bbox["image_id"]
            image_path = "{}/images/{}/{}.jpg".format(self.images_path, train_or_val, image_name)
            class_id = bbox["category_id"]
            class_name = list(filter(lambda c: c["id"] == class_id, self.classes))[0]["name"]
            x, y, w, h = bbox["bbox"]
            bbox_coords = [x, y, x + w, y + h]
            bbox_obj = {
                "image_path": image_path,
                "class_id": class_id,
                "class_name": class_name,
                "coords": bbox_coords,
            }
            self.bounding_boxes[image_name] = self.bounding_boxes.get(image_name, []) + [bbox_obj]
    
    def load_image(self, image_name):
        image_path = "{}/{}.jpg".format(self.images_path, image_name)
        image_ = imageio.imread(image_path).astype(np.uint8)[...,None]
        image = np.concatenate((image_, image_, image_), axis=2)    
        return image
    
    def __getitem__(self, idx):
        # load the image as a PIL Image
        image_name = self.images[idx]["id"]
        image = Image.fromarray(self.load_image(image_name))

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = [bbox["coords"] for bbox in self.bounding_boxes.get(image_name, [])]
        # and labels
        labels = torch.tensor([bbox["class_id"] for bbox in self.bounding_boxes.get(image_name, [])])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx
    
    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_height = self.images[idx]["height"]
        img_width = self.images[idx]["width"]
        return {"height": img_height, "width": img_width}
    
    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", dest="dataset_path", help="Path to the COCO dataset", required=True)
    args = parser.parse_args()
    
    print("[ Dataset Loading ... ]")
    Dataset = MIASDataset(args.dataset_path)
    print("[ OK ! ]")

    print("[ Dataset Length ... ]")
    length = len(Dataset)
    print(length)
    print("[ OK ! ]")