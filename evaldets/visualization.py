import copy
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Collection

import cv2
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np

from uo.utils import is_notebook, load
from jpeg import opencv_degrade_image
from .names import Names
from .results import DetectionResults


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
            k = itemgetter("image_id")
            self.image_objects = {
                key: list(value)
                for key, value in groupby(sorted(self.gt.values(), key=k), key=k)
            }
        return self.image_objects


DSI = DatasetIndex()
IMAGE_ROOT = Path(DSI.meta.image_root)


def image_for_id(image_id, quality=101):
    path = IMAGE_ROOT / f"{image_id:012d}.jpg"
    img = cv2.imread(str(path))
    return img


def visualizer_for_id(image_id, q=None, **kwargs):
    img = image_for_id(image_id)
    if q is not None:
        img = opencv_degrade_image(img, q)
    visualizer = Visualizer(img[:, :, ::-1], metadata=DSI.meta, **kwargs)
    return visualizer


def cv2_imshow(a, rgb=False):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a: np.ndarray.
            shape (N, M) or (N, M, 1) is an NxM grayscale image.
            shape (N, M, 3) is an NxM BGR color image.
            shape (N, M, 4) is an NxM BGRA color image.
        rgb: bool.
            `a` is (already) RGB instead of BGR.

    Return value:
        outside Jupyter: return the key pressed (see cv2.waitKey()).
        otherwise: None
    """
    import cv2

    if not is_notebook():
        if rgb:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", a)
        return cv2.waitKey(0)

    from PIL import Image
    from IPython.display import display

    a = a.clip(0, 255).astype("uint8")
    # cv2 stores colors as BGR; convert to RGB
    if not rgb and a.ndim == 3:
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
        cv2_imshow(v_img, True)


def draw_boxes(visualizer, boxes, labels):
    boxes = BoxMode.convert(np.array(boxes), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    vis = visualizer.overlay_instances(boxes=boxes, labels=labels)
    return vis.get_image()


def draw_box(visualizer, box, label):
    return draw_boxes(visualizer, [box], [label])


def show_image_objects(image_id, *, show_ids=True, category=None, scale=1.0, v=0):
    visualizer = visualizer_for_id(image_id, scale=scale)
    objects = DSI.gt_on_img[image_id]
    if category:
        objects = [obj for obj in objects if obj["category"] == category]

    if v:
        for ix, gt in enumerate(objects):
            print(ix + 1, gt)

    boxes = [obj["bbox"] for obj in objects]
    if show_ids:
        labels = [
            f'{obj["category"]} #{obj["id"]} {"(crowd)" if obj["iscrowd"] else ""}'
            for obj in objects
        ]
    else:
        labels = [obj["category"] for obj in objects]

    v_img = draw_boxes(visualizer, boxes, labels)
    cv2_imshow(v_img, True)


def _crop_detection(v_img: np.array, det: dict, margin=5, scale=1.0) -> np.array:
    x0, y0, w, h = itemgetter(*"xywh")(det)
    x1, y1 = x0 + w, y0 + h

    # crop around union (det, gt)
    if "gt_id" in det:
        gt = DSI.gt[det["gt_id"]]
        gx, gy, gw, gh = gt["bbox"]
        x0 = min(gx, x0)
        y0 = min(gy, y0)
        x1 = max(x1, gx + gw)
        y1 = max(y1, gy + gh)

    x0, y0, x1, y1 = (int(x * scale) for x in (x0, y0, x1, y1))

    return v_img[
        max(y0 - margin, 0) : y1 + margin,  # noqa
        max(x0 - margin, 0) : x1 + margin,  # noqa
        ...,
    ]


def show_detection(det: dict, *, crop=False, crop_margin=5, mode="cv2", q=None, scale=1.0, v=0):
    if crop is False:
        return show_detections([det], mode=mode, scale=scale, v=v)

    v_img = show_detections([det], mode="ret", scale=scale, v=v)
    v_img = _crop_detection(v_img, det, crop_margin, scale)

    if mode == "mpl":
        plt.imshow(v_img)
    elif mode == "cv2":
        return cv2_imshow(v_img, True)
    elif mode == "ret":
        return v_img


def show_detections(dets: Collection[dict], *, mode="cv2", q=None, scale=1.0, v=0):
    k = itemgetter("image_id")
    for image_id, img_dets in groupby(sorted(dets, key=k), key=k):
        print("image_id =", image_id)

        visualizer = visualizer_for_id(image_id, q, scale=scale)

        gt_boxes = []
        gt_labels = []
        boxes = []
        labels = []

        for det in img_dets:
            if "gt_id" in det:
                gt = DSI.gt[det["gt_id"]]
                gt_label = f"GT#{gt['id']}" + (" (crowd)" if gt["iscrowd"] else "")
                gt_boxes.append(gt["bbox"])
                gt_labels.append(gt_label)
            else:
                gt_label = "(no GT)"

            bbox = [det[k] for k in "xywh"]
            iou_label = f'J={det.get("iou", 0)*100:.1f}' if "gt_id" in det else "(FP)"
            label = f'{det["category"]} {det["score"]*100:.1f} {iou_label}'
            boxes.append(bbox)
            labels.append(label)

            if v:
                print(f'img={det["image_id"]}: {label} {gt_label}')

        if gt_boxes:
            draw_boxes(visualizer, gt_boxes, gt_labels)
        v_img = draw_boxes(visualizer, boxes, labels)

        if mode == "mpl":
            plt.imshow(v_img)
        elif mode == "cv2":
            key = cv2_imshow(v_img, True)
            if key == ord("q"):
                return
        elif mode == "ret":
            return v_img
        else:
            raise ValueError(f"Invalid mode for show_detections(): {mode}")


def browse_image(image_id: int):
    import webbrowser

    url = f"https://cocodataset.org/#explore?id={image_id}"
    webbrowser.open_new_tab(url)


def _main():
    import argparse

    parser = argparse.ArgumentParser("view_detections")
    parser.add_argument("detections_path")
    parser.add_argument("--scale", "-s", type=float, default=1.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--category", "-c")
    parser.add_argument("--image-id", "-i", type=int)
    args = parser.parse_args()

    dr = DetectionResults(args.detections_path)
    if args.image_id:
        detections = dr.detections_by_image_id(args.image_id)
        if args.category:
            detections = [d for d in detections if d["category"] == args.category]
        show_detections(detections, v=args.verbose, scale=args.scale)
    elif args.category:
        show_detections(
            dr.detections_by_class(args.category), v=args.verbose, scale=args.scale
        )
    else:
        show_detections(dr, v=args.verbose, scale=args.scale)


def _show_gt():
    import argparse

    parser = argparse.ArgumentParser("show_gt")
    parser.add_argument("image_id", type=int)
    parser.add_argument("--category", "-c")
    parser.add_argument("--scale", "-s", type=float, default=1.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--www", "-w", action="store_true")
    args = parser.parse_args()
    if args.www:
        browse_image(args.image_id)
    else:
        show_image_objects(
            args.image_id, category=args.category, scale=args.scale, v=args.verbose
        )


if __name__ == "__main__":
    _main()
