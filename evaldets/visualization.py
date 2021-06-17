import copy
import itertools
from operator import itemgetter
from pathlib import Path

import cv2
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np

from uo.utils import is_notebook, load
from .names import Names


def _load_gt_objects(meta):
    data = load(meta.json_file)
    names = Names(meta)
    anns = data["annotations"]
    result = {}
    for d in anns:
        del d["segmentation"]
        d["category"] = names.get(d["category_id"])
        del d["category_id"]
        result[d["id"]] = d
    return result


class DatasetIndex:
    def __init__(self, dataset="coco_2017_val") -> None:
        self.dataset = dataset
        self._meta = None
        self._names = None
        self.gt_objects = None
        self.image_objects = None

    @property
    def meta(self):
        if self._meta is None:
            self._meta = MetadataCatalog.get(self.dataset)
        return self._meta

    @property
    def gt(self):
        if self.gt_objects is None:
            self.gt_objects = _load_gt_objects(self.meta)
        return self.gt_objects

    @property
    def gt_on_img(self):
        if self.image_objects is None:
            key = itemgetter("image_id")
            self.image_objects = {
                key: list(value)
                for key, value in itertools.groupby(
                    sorted(self.gt.values(), key=key), key=key
                )
            }
        return self.image_objects


DSI = DatasetIndex()
IMAGE_ROOT = Path(DSI.meta.image_root)


def image_for_id(image_id):
    path = IMAGE_ROOT / f"{image_id:012d}.jpg"
    img = cv2.imread(str(path))
    return img


def visualizer_for_id(image_id, **kwargs):
    img = image_for_id(image_id)
    visualizer = Visualizer(img[:, :, ::-1], metadata=DSI.meta, **kwargs)
    return visualizer


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
    """Deprecated."""
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


def draw_boxes(visualizer, boxes, labels):
    boxes = BoxMode.convert(np.array(boxes), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    vis = visualizer.overlay_instances(boxes=boxes, labels=labels)
    return vis.get_image()


def draw_box(visualizer, box, label):
    return draw_boxes(visualizer, [box], [label])


def show_image_objects(image_id, *, show_ids=True):
    visualizer = visualizer_for_id(image_id)
    boxes = [obj["bbox"] for obj in DSI.gt_on_img[image_id]]
    if show_ids:
        labels = [f'{obj["category"]} #{obj["id"]}' for obj in DSI.gt_on_img[image_id]]
    else:
        labels = [obj["category"] for obj in DSI.gt_on_img[image_id]]
    v_img = draw_boxes(visualizer, boxes, labels)
    cv2_imshow(v_img[:, :, ::-1])


def show_image_detection(det: dict, mpl=False, scale=1.0, *, v=0):
    visualizer = visualizer_for_id(det["image_id"], scale=scale)

    if "gt_id" in det:
        # GT first, below detection
        gt = DSI.gt[det["gt_id"]]
        gt_label = f"GT#{gt['id']}" + (" (crowd)" if gt["iscrowd"] else "")
        draw_box(visualizer, gt["bbox"], gt_label)

    bbox = [det[k] for k in "xywh"]
    iou_label = f'J={det.get("iou", 0)*100:.1f}' if "gt_id" in det else "(FP)"
    label = f'{det["category"]} {det["score"]*100:.1f} {iou_label}'
    v_img = draw_box(visualizer, bbox, label)

    if v:
        print(f'img={det["image_id"]}: {label} {gt_label}')

    if mpl:
        plt.imshow(v_img)
    else:
        cv2_imshow(v_img[:, :, ::-1])
