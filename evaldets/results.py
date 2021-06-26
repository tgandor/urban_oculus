import glob
import os
from operator import itemgetter
import pickle

import numpy as np
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from uo.utils import load

from .metrics import interpolated_PPV
from .names import Names


def load_gt(dataset="coco_2017_val", del_mask=True, debug=None):
    path = MetadataCatalog.get(dataset).json_file
    if not os.path.exists(path):
        print(f"Missing annotations JSON: {path}")
        exit()
    kwargs = {}
    if debug is not None:
        kwargs['debug'] = debug
    result = COCO(path, **kwargs)

    if del_mask:
        for ann in result.dataset["annotations"]:
            del ann["segmentation"]

    return result


def load_detections(file_or_dir, cache=True):
    file_or_dir = os.path.expanduser(file_or_dir)

    if os.path.isdir(file_or_dir):
        dump_dir = file_or_dir
        det_filename = glob.glob(
            os.path.join(dump_dir, "coco_instances_results.json*")
        )[0]
    else:
        dump_dir = os.path.dirname(file_or_dir)
        det_filename = file_or_dir

    cache_file = os.path.join(dump_dir, "detections.pkl")
    if cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as pkl:
            detections = pickle.load(pkl)
        print('Loaded cached detections:', cache_file)
        return detections, cache_file

    detections = load(det_filename)

    return detections, det_filename


class DetectionResults:
    """
    Attributes:
        debug: int debug level of COCO and COCOeval
            This only works with my fork:
            https://github.com/tgandor/cocoapi/tree/reformatted
    """

    def __init__(
        self,
        det_file_or_dir,
        *,
        dataset="coco_2017_val",
        rounding=False,
        use_cats=True,
        area_rng=False,
        iou_thresh=0.5,
        debug=0,
        cache=True,
    ):
        self.dataset = dataset
        self.input = det_file_or_dir
        self.rounding = rounding
        self.debug = debug
        self.cache = cache
        self.cache_loaded = False
        # COCOeval params
        self.use_cats = use_cats
        self.area_rng = area_rng
        self.iou_thresh = iou_thresh
        # process
        self.names = Names.for_dataset(self.dataset)
        self._detections_by_image_id = None  # TODO: memoize
        self._detections_by_class = None  # TODO: memoize
        self._evaluate()
        self._enrich_detections()
        if cache and not self.cache_loaded:
            self._save_cache()

    def _evaluate(self):
        self.detections, self.det_file = load_detections(self.input, self.cache)
        self.gt = load_gt(self.dataset, debug=self.debug)

        if self.det_file.endswith('.pkl'):
            self.dt = None
            self.coco = None
            self.cache_loaded = True
            return

        self.dt = self.gt.loadRes(self.detections)

        kwargs = {}
        if self.debug is not None:
            kwargs['debug'] = self.debug
        self.coco = COCOeval(self.gt, self.dt, iouType="bbox", **kwargs)

        # don't evalImage for 'small', 'medium', 'large'
        if self.area_rng is False:
            self.coco.params.areaRng = [[0.0, 1e9]]
        elif self.area_rng is not None:
            self.coco.params.areaRng = self.area_rng

        # do or don't segregate objects by category
        self.coco.params.useCats = self.use_cats

        # set IoU threshold(s)
        if self.iou_thresh is not None:
            self.coco.params.iouThrs = [self.iou_thresh]

        self.coco.evaluate()

        nCats = len(self.coco.params.catIds) if self.use_cats else 1
        nArea = len(self.coco.params.areaRng)
        nImgs = len(self.coco.params.imgIds)

        # print("nCats:", nCats, len(self.coco.evalImgs), nCats * nArea * nImgs)
        assert len(self.coco.evalImgs) == nCats * nArea * nImgs

        for catIx in range(nCats):
            catId = self.coco.params.catIds[catIx] if self.use_cats else -1
            offs = catIx * (nArea * nImgs)
            for imgIx, img in enumerate(
                self.coco.evalImgs[offs : offs + nImgs]  # noqa
            ):
                if img is None:
                    continue
                imgId = self.coco.params.imgIds[imgIx]
                ious = self.coco.ious[imgId, catId]
                dt_id2dind = {dt_id: dind for dind, dt_id in enumerate(img["dtIds"])}
                for gind, fdt_id in enumerate(img["gtMatches"][0]):
                    dt_id = int(fdt_id)
                    if not dt_id:
                        continue
                    detection = self.detections[dt_id - 1]
                    dind = dt_id2dind[dt_id]  # per-image detection idx
                    gt_id = img["gtIds"][gind]
                    # detection["true_positive"] = True
                    detection["iou"] = ious[dind, gind]
                    detection["gt_id"] = gt_id
                    # annotation = self.gt.anns[gt_id]
                    # detection["gt_bbox"] = annotation["bbox"]

    def _enrich_detections(self):
        if self.cache_loaded:
            return

        for d in self.detections:
            d: dict
            del d["segmentation"]
            del d["id"]
            if d["iscrowd"]:
                # you should never see this:
                print('Found crowd:', d)
            del d["iscrowd"]

            # d.setdefault("true_positive", False)
            if self.rounding:
                d["score"] = round(d["score"], 5)
                d["bbox"] = [round(x, 1) for x in d["bbox"]]
                d["area"] = round(d["area"], 1)
                if "iou" in d:
                    d["iou"] = round(d["iou"], 3)

            # replace category_id with category name, as last key:
            d["category"] = self.names.get(d["category_id"])
            del d["category_id"]
            del d["area"]  # purely derivative from w*h (mod rounding)
            d["x"], d["y"], d["w"], d["h"] = d["bbox"]
            del d["bbox"]

    def _save_cache(self):
        cache_file = os.path.join(os.path.dirname(self.det_file), "detections.pkl")
        with open(cache_file, 'wb') as pkl:
            print('Saving detections to cache:', cache_file)
            pickle.dump(self.detections, pkl)

    def __iter__(self):
        return iter(self.detections)

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, idx):
        return self.detections[idx]

    @property
    def num_gt_all(self):
        return sum(g["ignore"] == 0 for v in self.coco._gts.values() for g in v)

    @property
    def detections_by_score(self):
        # sorted() is stable, so np.argsort(kind="mergesort") is no issue
        return sorted(self.detections, key=itemgetter("score"), reverse=True)

    def detections_by_class(self, name: str) -> list:
        # TODO: dict, sorted by score
        return [d for d in self.detections_by_score if d["category"] == name]

    def detections_by_image_id(self, image_id: int) -> list:
        # TODO: dict
        return [d for d in self.detections if d["image_id"] == image_id]

    def num_gt_class(self, name):
        cls_id = self.names.name_to_id(name)
        return sum(
            g["ignore"] == 0
            for (_, cat_id), v in self.coco._gts.items()
            for g in v
            if cat_id == cls_id
        )

    def average_precision(self, category: str, t_iou: float = 0.5):
        dets = self.detections_by_class(category)
        TP = np.cumsum([det.get('iou', 0) >= t_iou for det in dets])
        FP = np.cumsum([det.get('iou', 0) < t_iou for det in dets])
        nGT = self.num_gt_class(category)
        TPR = TP / nGT
        PPV = TP / (TP + FP)
        PPVi = interpolated_PPV(PPV)
        recThrs = self.coco.params.recThrs
        inds = np.searchsorted(TPR, recThrs, side="left")
        q = np.zeros_like(recThrs)
        for ri, pi in enumerate(inds):
            if pi >= len(PPVi):
                break
            q[ri] = PPVi[pi]
        return np.mean(q)

    def mean_average_precision(self, t_iou: float = 0.5):
        """mAP metric."""
        return np.mean([self.average_precision(c, t_iou) for c in self.names.all])
