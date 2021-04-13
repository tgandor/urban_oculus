import bz2
import copy
import gzip
import json

from collections import Counter

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import (
    build_detection_test_loader,
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.data.catalog import Metadata
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer

import matplotlib.pyplot as plt
import pandas as pd

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


def conf(model: str) -> CfgNode:
    model_config = MODEL_ZOO_CONFIGS[model]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    return cfg


def load(path: str) -> dict:
    if path.endswith('.json.gz'):
        with gzip.open(path) as fs:
            return json.load(fs)
    if path.endswith('.json.bz2'):
        with bz2.open(path) as fs:
            return json.load(fs)
    if path.endswith('.json'):
        with open(path) as fs:
            return json.load(fs)
    raise ValueError('unknown file type')


def is_notebook():
    # https://stackoverflow.com/a/39662359/1338797
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
      a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image.
                      shape (N, M, 3) is an NxM BGR color image.
                      shape (N, M, 4) is an NxM BGRA color image.
    """
    import cv2

    if not is_notebook():
        cv2.imshow('image', a)
        cv2.waitKey(0)
        return

    from PIL import Image
    from IPython.display import display

    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))


def show_image_gt(d: dict, meta: Metadata, mpl=False, no_mask=True) -> None:
    import cv2
    img = cv2.imread(d["file_name"])

    if no_mask:
        d = copy.deepcopy(d)
        for a in d['annotations']:
            if 'segmentation' in a:
                del a['segmentation']

    visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=1.0)
    vis = visualizer.draw_dataset_dict(d)
    v_img = vis.get_image()

    if mpl:
        plt.imshow(v_img)
    else:
        cv2_imshow(v_img[:, :, ::-1])
