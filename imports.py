import json
import bz2
from collections import Counter

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer

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

def conf(model):
    model_config = MODEL_ZOO_CONFIGS[model]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    return cfg

def load(path: str):
    if path.endswith('.json.bz2'):
        with bz2.open(path) as fs:
            return json.load(fs)
    if path.endswith('.json'):
        with open(path) as fs:
            return json.load(fs)
    raise ValueError('unknown file type')
