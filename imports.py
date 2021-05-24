import bz2
import copy
import gzip
import json

from collections import Counter  # noqa

from pycocotools.coco import COCO  # noqa
from pycocotools.cocoeval import COCOeval  # noqa

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import (  # noqa
    build_detection_test_loader,
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.data.catalog import Metadata
from detectron2.engine import DefaultPredictor  # noqa
from detectron2.evaluation import COCOEvaluator, inference_on_dataset  # noqa
from detectron2.utils.visualizer import Visualizer

import matplotlib.pyplot as plt
import pandas as pd  # noqa

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
    if path.endswith(".json.gz"):
        with gzip.open(path) as fs:
            return json.load(fs)
    if path.endswith(".json.bz2"):
        with bz2.open(path) as fs:
            return json.load(fs)
    if path.endswith(".json"):
        with open(path) as fs:
            return json.load(fs)
    raise ValueError("unknown file type")


def is_notebook():
    # https://stackoverflow.com/a/39662359/1338797
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
      a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image.
                      shape (N, M, 3) is an NxM BGR color image.
                      shape (N, M, 4) is an NxM BGRA color image.
    """
    import cv2

    if not is_notebook():
        cv2.imshow("image", a)
        cv2.waitKey(0)
        return

    from PIL import Image
    from IPython.display import display

    a = a.clip(0, 255).astype("uint8")
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
        for a in d["annotations"]:
            if "segmentation" in a:
                del a["segmentation"]

    visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=1.0)
    vis = visualizer.draw_dataset_dict(d)
    v_img = vis.get_image()

    if mpl:
        plt.imshow(v_img)
    else:
        cv2_imshow(v_img[:, :, ::-1])


class Names:
    def __init__(self, meta: Metadata) -> None:
        self.meta = meta
        self.idx_to_id = {
            v: k for k, v in self.meta.thing_dataset_id_to_contiguous_id.items()
        }
        self.name_to_i = {v: i for i, v in enumerate(self.meta.thing_classes)}

    def get(self, id: int) -> str:
        return self.meta.thing_classes[self.meta.thing_dataset_id_to_contiguous_id[id]]

    def name_to_id(self, name):
        return self.idx_to_id.get(self.name_to_idx(name))

    def name_to_idx(self, name):
        return self.name_to_i.get(name)


def show_image_detections():
    ...
