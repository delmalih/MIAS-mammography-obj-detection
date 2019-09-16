###########
# Imports #
###########

""" Global """

import keras
import tensorflow as tf
import numpy as np
import cv2
import argparse, os, glob, subprocess
from keras_retinanet import models
from keras_retinanet import preprocessing

""" Local """

# from datagenerator import DataGenerator
# from model import Model

##############
# Parameters #
##############

n_classes = 1
n_epochs = 500
image_size = 512
batch_size = 64
learning_rate = 0.001
weights_path = "model.hdf5"
tmp_dataset_folder = "/tmp/dataset"

#############
# Functions #
#############

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Train the model for shelves detection")
    parser.add_argument('-d', '--dataset', dest='dataset_path',
                        help="Path of the COCO dataset", required=True)
    parser.add_argument('--backbone', help='Backbone model used by retinanet.', default='resnet50', type=str)
    return parser.parse_args()

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def create_generators(dataset_path):
    train_generator = preprocessing.coco.CocoGenerator(dataset_path, 'train')

    validation_generator = preprocessing.coco.CocoGenerator(dataset_path, 'val')

    return train_generator, validation_generator

def train(args, train_generator, val_generator):
    backbone = models.backbone(args.backbone)
    keras.backend.tensorflow_backend.set_session(get_session())
    train_generator, validation_generator = create_generators(args.dataset_path)

    tf_logdir = "/tmp/retinanet"
    subprocess.Popen(['sudo', 'rm', '-rf', tf_logdir])
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    tensorBoard = keras.callbacks.TensorBoard(log_dir=tf_logdir,
                                              histogram_freq=1,
                                              batch_size=batch_size,
                                              write_grads=True,
                                              write_images=True,
                                              update_freq="batch")
    
    model = keras_retinanet.models.backbone('resnet50').retinanet(num_classes=n_classes)
    model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss="binary_crossentropy", metrics=[iou, "accuracy"])
    
    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        epochs=n_epochs, verbose=1,
                        callbacks=[tensorBoard, checkpointer])

def run(args):
    global image_size, weights_path, n_channels, n_classes, batch_size
    dataset_path = args["dataset_path"]
    multiply_by = int(args["multiply_by"])
    validation_size = float(args["validation_size"]) if float(args["validation_size"]) < 1 else 1.0
    if not args["keep_database"]: generate_train_test_folders(dataset_path, multiply_by, validation_size)
    model = Model(image_size, image_size, weights_path, n_channels, n_classes)
    train_generator = get_generator("{}/train/images/*".format(tmp_dataset_folder), "{}/train/labels/*".format(tmp_dataset_folder), n_classes, batch_size=batch_size, n_repeat=len(model.outputs))
    val_generator = get_generator("{}/val/images/*".format(tmp_dataset_folder), "{}/val/labels/*".format(tmp_dataset_folder), n_classes, batch_size=batch_size, n_repeat=len(model.outputs))
    train(model, train_generator, val_generator)

if __name__ == '__main__':
    args = parse_args()
    args = vars(args)
    run(args)