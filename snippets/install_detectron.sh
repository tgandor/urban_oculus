#!/bin/bash

if [ ! -f DET2_INSTALLED ] ; then
    time pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    time python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    time python -m pip install -U pyyaml
    touch DET2_INSTALLED
else
    echo "Detectron seems to have been installed: remove DET2_INSTALLED to force."
fi
