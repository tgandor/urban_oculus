#!/bin/bash

time pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
time python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
time python -m pip install -U pyyaml
