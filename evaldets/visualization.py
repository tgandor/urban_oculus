import copy
from pathlib import Path

import cv2
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.structures import BoxMode  # noqa
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np

from uo.utils import is_notebook, load


def _load_gt_objects(meta):
    data = load(meta.json_file)
    anns = data["annotations"]
    result = {}
    for d in anns:
        del d["segmentation"]
        result[d["id"]] = d
    return result


META = MetadataCatalog.get("coco_2017_val")
GT = _load_gt_objects(META)
IMAGE_ROOT = Path(META.image_root)


def image_for_id(image_id):
    path = IMAGE_ROOT / f"{image_id:012d}.jpg"
    img = cv2.imread(str(path))
    return img


def visualizer_for_id(image_id, **kwargs):
    img = image_for_id(image_id)
    visualizer = Visualizer(img[:, :, ::-1], metadata=META, **kwargs)
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


def draw_box(visualizer, box, label):
    boxes = [box]
    labels = [label]
    boxes = BoxMode.convert(np.array(boxes), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    vis = visualizer.overlay_instances(boxes=boxes, labels=labels)
    return vis.get_image()


def show_image_detection(det: dict, mpl=False, scale=1.0, *, v=0):
    visualizer = visualizer_for_id(det["image_id"], scale=scale)

    if "gt_id" in det:
        # GT first, below detection
        gt = GT[det["gt_id"]]
        gt_label = f"GT#{gt['id']}" + (" (crowd)" if gt["iscrowd"] else "")
        draw_box(visualizer, gt['bbox'], gt_label)

    bbox = [det[k] for k in "xywh"]
    iou_label = f'IoU={det.get("iou", 0):.3f}' if "gt_id" in det else "(FP)"
    label = f'{det["category"]} p={det["score"]:.3f} {iou_label}'
    v_img = draw_box(visualizer, bbox, label)

    if v:
        print(f'img={det["image_id"]}: {label} {gt_label}')

    if mpl:
        plt.imshow(v_img)
    else:
        cv2_imshow(v_img[:, :, ::-1])
