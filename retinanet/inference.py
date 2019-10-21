###########
# Imports #
###########

import argparse
import os
import sys
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf

from keras_retinanet import layers  # noqa: F401
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.preprocessing.kitti import KittiGenerator
from keras_retinanet.preprocessing.open_images import OpenImagesGenerator
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.image import random_visual_effect_generator

#############
# Functions #
#############


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple inference script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    
    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--set_name',         help='Name of the set containing the images to infere', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')

    return check_args(parser.parse_args(args))


def create_generators(args):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
    }

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from keras_retinanet.preprocessing.coco import CocoGenerator
        generator = CocoGenerator(
            args.coco_path,
            args.set_name,
            **common_args
        )
    elif args.dataset_type == 'pascal':
        generator = PascalVocGenerator(
            args.pascal_path,
            args.set_name,
            **common_args
        )
    elif args.dataset_type == 'csv':
        generator = CSVGenerator(
            args.annotations,
            args.classes,
            **common_args
        )
    elif args.dataset_type == 'oid':
        generator = OpenImagesGenerator(
            args.main_dir,
            subset=args.set_name,
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            parent_label=args.parent_label,
            **common_args
        )
    elif args.dataset_type == 'kitti':
        generator = KittiGenerator(
            args.kitti_path,
            subset=args.set_name,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generators
    generator = create_generators(args)

    # create the model
    print('Loading model, this may take a second...')
    model            = models.load_model(args.snapshot, backbone_name=args.backbone)
    anchor_params    = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # print model summary
    prediction_model.summary()

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        generator.compute_shapes = make_shapes_callback(model)

    # inference
    boxes, classification, _ = prediction_model.predict_generator(
        generator=generator,
        verbose=1,
    )

    print(boxes)


if __name__ == '__main__':
    main()