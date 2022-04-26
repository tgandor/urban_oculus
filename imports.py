# flake8: noqa
import bz2
import copy
import gzip
import json

from collections import Counter  # noqa
from itertools import islice

from pycocotools.coco import COCO  # noqa
from pycocotools.cocoeval import COCOeval  # noqa

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data.catalog import Metadata
from detectron2.data import (  # noqa
    build_detection_test_loader,
    DatasetCatalog,
    MetadataCatalog,
)

from detectron2.engine import DefaultPredictor  # noqa
from detectron2.evaluation import COCOEvaluator, inference_on_dataset  # noqa
from detectron2.structures import BoxMode  # noqa
from detectron2.utils.visualizer import Visualizer

import matplotlib.pyplot as plt
import pandas as pd  # noqa

from uo.utils import is_notebook, load, top
from evaldets.api import *

MODEL_ZOO_CONFIGS = {
    "R50_C4": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
    "R50_DC5": "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
    "R50_FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "R101_C4": "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
    "R101_DC5": "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
    "R101_FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "X101": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "R50": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    "R101": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
}

CONFIGS_1x = {
    "R50_C4": "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
    "R50_DC5": "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
    "R50_FPN": "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
    "R50": "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
}


def conf(model: str, configs=None) -> CfgNode:
    import warnings

    if configs is None:
        configs = MODEL_ZOO_CONFIGS
    model_config = configs[model]
    cfg = get_cfg()
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore")
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
    return cfg
