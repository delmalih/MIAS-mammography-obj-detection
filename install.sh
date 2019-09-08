# Considering conda is already installed and you are in you environment

conda install ipython

pip install -r requirements.txt

git clone https://github.com/tensorflow/models tf-models

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cd ../../
cp -r pycocotools tf-models/research/

brew install protobuf
cd tf-models/research/
protoc object_detection/protos/*.proto --python_out=.

cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/research:`pwd`/research/slim

python research/object_detection/builders/model_builder_test.py
