# Breast cancer tumor detection on mammograms

## Requirements

- GCC >= 4.9
- CUDA 9.0 & cuDNN 7.0 ([install. instructions](https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8#install-nvidia-graphics-driver-via-apt-get))
- Anaconda 3 ([install. instructions](https://problemsolvingwithpython.com/01-Orientation/01.05-Installing-Anaconda-on-Linux/))

## References

- Faster-RCNN paper: [arxiv.org/pdf/1506.01497.pdf](https://arxiv.org/pdf/1506.01497.pdf)
- Faster-RCNN implem. repo.: [github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- RetinaNet paper: [arxiv.org/pdf/1708.02002.pdf](https://arxiv.org/pdf/1708.02002.pdf)
- RetinaNet implem. repo.: [github.com/fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
- FCOS paper: [arxiv.org/pdf/1904.01355.pdf](https://arxiv.org/pdf/1904.01355.pdf)
- FCOS implem. repo.: [github.com/tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS)

## Installation instructions

Start by cloning this repo:

```
git clone https://github.com/delmalih/MIAS-mammography-obj-detection
```

### 1. Faster R-CNN instructions

- First, create an environment :

```
conda create --name faster-r-cnn
conda activate faster-r-cnn
conda install ipython pip
cd mias-mammography-obj-detection
pip install -r requirements.txt
cd ..
```

- Then, run these commands (ignore if you have already done the FCOS installation) :

```
# install pytorch
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
python setup.py build develop

cd $INSTALL_DIR
unset INSTALL_DIR
```

### 2. RetinaNet instructions

- First, create an environment :

```
conda create --name retinanet
conda activate retinanet
conda install ipython pip
pip install -r requirements.txt
```

- Then, run these commands :

```
# clone keras-retinanet repo
git clone https://github.com/fizyr/keras-retinanet
pip install . --user
python setup.py build_ext --inplace

# install pycoco tools
pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

- Finally, replace the `keras_retinanet/preprocessing/coco.py` file by [this file](https://github.com/delmalih/mias-mammography-obj-detection/blob/master/retinanet/coco.py)

### 3. FCOS instructions

- First, create an environment :

```
conda create --name fcos
conda activate fcos
conda install ipython pip
pip install -r requirements.txt
cd fcos
```

- Then, follow [these instructions](https://github.com/tianzhi0549/FCOS#installation)

## How it works

### 1. Download the MIAS Database

Run these commands to download to MIAS database :

```
mkdir mias-db && cd mias-db
wget http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz
tar -zxvf all-mias.tar.gz
rm all-mias.tar.gz && cd ..
```

And replace the `mias-db/Info.txt` by [this one](https://raw.githubusercontent.com/delmalih/MIAS-mammography-obj-detection/master/utils/Info.txt)

### 2. Generate COCO or VOC augmented data

It is possible to generate COCO or VOC annotations from raw data (`all-mias` folder + `Info.txt` annotations file) through 2 scripts: `generate_{COCO|VOC}_annotations.py` :

```
python generate_{COCO|VOC}_annotations.py --images (or -i) <Path to the images folder> \
                                          --annotations (or -a) <Path to the .txt annotations file> \
                                          --output (or -o) <Path to output folder> \
                                          --aug_fact <Data augmentation factor> \
                                          --train_val_split <Percetange of the train folder (default 0.9)>
```

For example, to generate 10x augmented COCO annotations, run this command :

```
python generate_COCO_annotations.py --images ../mias-db/ \
                                    --annotations ../mias-db/Info.txt \
                                    --output ../mias-db/COCO \
                                    --aug_fact 20 \
                                    --train_val_split 0.9
```

### 3. How to run a training

#### 3.1 Faster R-CNN

To run a training with the Faster-RCNN:

- Follow [these instructions](https://github.com/facebookresearch/maskrcnn-benchmark#adding-your-own-dataset)
- Run this command :

```
cd faster-r-cnn
conda deactivate && conda activate faster-r-cnn
python train.py --config-file <Path to the config file>
```

#### 3.2 RetinaNet

To run a training with the retinanet :

```
cd retinanet
conda deactivate && conda activate retinanet
python train.py --compute-val-loss \ # Computer val loss or not
                --tensorboard-dir <Path to the tensorboard directory> \
                --batch-size <Batch size> \
                --epochs <Nb of epochs> \
                coco <Path to the COCO dataset>
```

And if you want to see the tensorboard, run on another window :

```
tensorboard --logdir <Path to the tensorboard directory>
```

#### 3.3 FCOS

To run a training with the FCOS Object Detector :

- Follow [these instructions](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687)
- Run this command :

```
cd fcos
conda deactivate && conda activate fcos
python train.py --config-file <Path to the config file> \
                OUTPUT_DIR <Path to the output dir for the logs>
```

### 4. How to run an inference

#### 4.1 Faster R-CNN

```
cd faster-r-cnn
conda deactivate && conda activate faster-r-cnn
python inference.py --config-file <Path to the config file> \
                    MODEL.WEIGHT <Path to weights of the model to load> \
                    TEST.IMS_PER_BATCH <Nb of images per batch>
```


#### 4.2 RetinaNet

- Put the images you want to run an inference on, in `<Name of COCO dataset>/<Name of folder>`
- Run this command :

```
cd retinanet
conda deactivate && conda activate retinanet
python inference.py --snapshot <Path of the model snapshot> \
                    --set_name <Name of the inference folder in the COCO dataset> \
                    coco <Path to the COCO dataset>
```

#### 4.3 FCOS

```
cd fcos
conda deactivate && conda activate fcos
python inference.py --config-file <Path to the config file> \
                    MODEL.WEIGHT <Path to weights of the model to load> \
                    TEST.IMS_PER_BATCH <Nb of images per batch>
```
