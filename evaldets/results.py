from collections import defaultdict
import glob
import os
import time
from itertools import groupby
from operator import itemgetter

import numpy as np
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from uo.utils import aspectize, load, logged, save

from .metrics import interpolated_PPV
from .names import Names

CROWD_ID_T = 10 ** 9
"""
In [8]: '{:,}'.format(max(a['id'] for a in dr.gt.dataset['annotations']))
Out[8]: '908,800,474,293'

In [9]: '{:,}'.format(min(a['id'] for a in dr.gt.dataset['annotations'] if a['id'] > 1e9))
Out[9]: '900,100,002,299'

In [10]: '{:,}'.format(max(a['id'] for a in dr.gt.dataset['annotations'] if a['id'] < 1e9))
Out[10]: '2,232,119'

In [11]: '{:,}'.format(min(a['id'] for a in dr.gt.dataset['annotations']))
Out[11]: '283'
"""


def load_gt(dataset="coco_2017_val", del_mask=True, debug=None):
    path = MetadataCatalog.get(dataset).json_file
    if not os.path.exists(path):
        print(f"Missing annotations JSON: {path}")
        exit()
    kwargs = {}
    if debug is not None:
        kwargs["debug"] = debug
    result = COCO(path, **kwargs)

    if del_mask:
        for ann in result.dataset["annotations"]:
            del ann["segmentation"]

    return result


def load_detections(file_or_dir, cache=True, debug=True):
    file_or_dir = os.path.expanduser(file_or_dir)

    if os.path.isdir(file_or_dir):
        dump_dir = file_or_dir
        det_filename = glob.glob(
            os.path.join(dump_dir, "coco_instances_results.json*")
        )[0]
    elif os.path.exists(file_or_dir):
        dump_dir = os.path.dirname(file_or_dir)
        det_filename = file_or_dir
    else:
        raise ValueError(f"File or directory not found: {file_or_dir}")

    cache_file = os.path.join(dump_dir, "detections.pkl")
    if cache and os.path.exists(cache_file):
        detections = load(cache_file)
        if debug:
            print("Loaded cached detections:", cache_file)
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

    # recThrs = self.coco.params.recThrs
    RECALL_THRS = np.linspace(
        0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
    )
    # coco.params.iouThrs
    IOU_THRS = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )
    # coco.params.maxDet = [1, 10, 100]
    TOP_MAX_DET = -1
    # coco.params.areaRng = [<all>, <small>, <medium>, <large>]
    ALL_AREA = 0

    def __init__(
        self,
        det_file_or_dir,
        *,
        area_rng=False,
        cache=True,
        dataset="coco_2017_val",
        debug=0,
        gt_match=False,
        iou_thresh=0.5,
        rounding=False,
        use_cats=True,
    ):
        self.cache = cache
        self.dataset = dataset
        self.debug = debug
        self.gt_match = gt_match
        self.input = det_file_or_dir
        self.rounding = rounding
        # COCOeval params
        self.area_rng = area_rng
        self.iou_thresh = iou_thresh
        self.use_cats = use_cats
        # process
        self.names = Names.for_dataset(self.dataset)
        self._detections_by_image_id = None  # TODO: memoize
        self._detections_by_class = None
        self._all_detections_by_class = None
        self.gt = load_gt(self.dataset, debug=self.debug)

        k = itemgetter("category_id")
        self._num_gt_class = {
            self.names.get(cid): sum(g["iscrowd"] == 0 for g in values)
            for cid, values in groupby(sorted(self.gt.anns.values(), key=k), key=k)
        }
        self._num_gt_all = sum(self._num_gt_class.values())

        self._load_detections()

        if not self.cache_loaded:
            self._evaluate()
            self._match_detections()
            self._enrich_detections()
            if cache:
                self._save_cache()

    def _load_detections(self, no_cache: bool = False):
        self.detections, self.det_file = load_detections(
            self.input, self.cache and not no_cache, self.debug
        )

        if self.det_file.endswith(".pkl"):
            self.dt = None
            self.coco = None
            self.cache_loaded = True
        else:
            self.cache_loaded = False

    def _evaluate(self):
        cache_file = os.path.join(os.path.dirname(self.det_file), "coco.pkl.gz")
        if self.cache and os.path.exists(cache_file):
            self.coco = load(cache_file)
            print(f"Loaded cached COCOEval: {cache_file}")
            return

        self.dt = self.gt.loadRes(self.detections)

        kwargs = {}
        if self.debug is not None:
            kwargs["debug"] = self.debug
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

    def _reset_detections(self):
        for detection in self.detections:
            if "gt_id" in detection:
                del detection["gt_id"]
            if "iou" in detection:
                del detection["iou"]
        self._all_detections_by_class = None

    def _match_detections(self, iou_index=0):
        def _match_by_gt(img):
            image_id = img["image_id"]
            category_id = img["category_id"] if self.use_cats else -1
            ious = self.coco.ious[image_id, category_id]
            dt_id2dind = {dt_id: dind for dind, dt_id in enumerate(img["dtIds"])}
            for gind, fdt_id in enumerate(img["gtMatches"][iou_index]):
                dt_id = int(fdt_id)
                if not dt_id:
                    continue
                detection = self.detections[dt_id - 1]
                dind = dt_id2dind[dt_id]  # per-image detection idx
                detection["iou"] = ious[dind, gind]
                detection["gt_id"] = img["gtIds"][gind]

        def _match_by_dt(img):
            image_id = img["image_id"]
            category_id = img["category_id"] if self.use_cats else -1
            ious = self.coco.ious[image_id, category_id]
            gt_id2gind = {gt_id: gind for gind, gt_id in enumerate(img["gtIds"])}
            for dind, fgt_id in enumerate(img["dtMatches"][iou_index]):
                gt_id = int(fgt_id)
                if not gt_id:
                    continue
                dt_id = img["dtIds"][dind]
                detection = self.detections[dt_id - 1]
                gind = gt_id2gind[gt_id]
                detection["iou"] = ious[dind, gind]
                detection["gt_id"] = img["gtIds"][gind]

        matching_strategy = _match_by_gt if self.gt_match else _match_by_dt

        nCats = len(self.coco.params.catIds) if self.use_cats else 1
        nArea = len(self.coco.params.areaRng)
        nImgs = len(self.coco.params.imgIds)

        assert len(self.coco.evalImgs) == nCats * nArea * nImgs

        for catIx in range(nCats):
            offs = catIx * (nArea * nImgs)
            for img in filter(None, self.coco.evalImgs[offs : offs + nImgs]):  # noqa
                matching_strategy(img)

    def _enrich_detections(self):
        for d in self.detections:
            d: dict
            if "segmentation" in d:
                del d["segmentation"]

            if "iscrowd" in d:
                if d["iscrowd"]:
                    # you should never see this:
                    print("Found crowd:", d)
                del d["iscrowd"]

            if "category_id" in d:
                # replace category_id with category name
                d["category"] = self.names.get(d["category_id"])

            if "bbox" in d:
                d["x"], d["y"], d["w"], d["h"] = d["bbox"]

    def _clean_detections(self):
        for d in self.detections:
            d: dict
            if "segmentation" in d:
                del d["segmentation"]
            if "id" in d:
                del d["id"]
            if "iscrowd" in d:
                if d["iscrowd"]:
                    # you should never see this:
                    print("Found crowd:", d)
                del d["iscrowd"]

            if self.rounding:
                d["score"] = round(d["score"], 5)
                d["bbox"] = [round(x, 1) for x in d["bbox"]]
                d["area"] = round(d["area"], 1)
                if "iou" in d:
                    d["iou"] = round(d["iou"], 3)

            if "category_id" in d:
                del d["category_id"]
            if "area" in d:
                del d["area"]  # derivative from w*h
            if "bbox" in d:
                del d["bbox"]

    def _ensure_cocoeval(self):
        if self.coco is None:
            # COCO params for compatibility
            self.area_rng = None
            self.iou_thresh = None
            self._evaluate()

    def match_detections(self, iou_index=0):
        self._ensure_cocoeval()
        self._reset_detections()
        self._match_detections(iou_index)
        self._enrich_detections()

    def finish_cocoeval(self):
        self._ensure_cocoeval()
        self.coco.accumulate()
        self.coco.summarize()

    def _save_cache(self):
        cache_file = os.path.join(os.path.dirname(self.det_file), "detections.pkl")
        print("Saving detections to cache:", cache_file)
        save(self.detections, cache_file)

    def save_cocoeval(self):
        """Save the COCOeval object for future re-use."""
        cache_file = os.path.join(os.path.dirname(self.det_file), "coco.pkl.gz")
        save(self.coco, cache_file)

    def __iter__(self):
        return iter(self.detections)

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, idx):
        return self.detections[idx]

    def __repr__(self) -> str:
        return "<DetectionResult>"

    @property
    def detections_by_score(self):
        # sorted() is stable, so np.argsort(kind="mergesort") is no issue
        return sorted(self.detections, key=itemgetter("score"), reverse=True)

    def all_detections_by_class(self, name: str) -> list:
        if self._all_detections_by_class is None:
            self._all_detections_by_class = {
                category: list(detections)
                for category, detections in groupby(
                    sorted(
                        self.detections,
                        key=itemgetter("category", "score"),
                        reverse=True,
                    ),
                    key=itemgetter("category"),
                )
            }

        return self._all_detections_by_class.get(name, [])

    def detections_by_class(self, name: str) -> list:
        if self._detections_by_class is None:
            self._detections_by_class = {
                cat: [
                    d
                    for d in self.all_detections_by_class(cat)
                    if d.get("gt_id", 0) < CROWD_ID_T
                ]
                for cat in self.names
            }

        return self._detections_by_class[name]

    def detections_by_image_id(self, image_id: int) -> list:
        # TODO: dict
        return [d for d in self.detections if d["image_id"] == image_id]

    @property
    def num_gt_all(self):
        return self._num_gt_all

    def num_gt_class(self, name):
        return self._num_gt_class[name]

    def _tp_sum(self, category: str, t_iou: float):
        return np.cumsum(
            [
                (det.get("iou", 0) >= t_iou and det.get("gt_id", 0) < CROWD_ID_T)
                for det in self.all_detections_by_class(category)
            ]
        ).astype(float)

    def _fp_sum(self, category: str, t_iou: float):
        return np.cumsum(
            [
                (det.get("iou", 0) < t_iou and det.get("gt_id", 0) < CROWD_ID_T)
                for det in self.all_detections_by_class(category)
            ]
        ).astype(float)

    def tp_sum_all(self, t_iou: float = 0.5):
        """Return sum of """
        return np.cumsum(
            [
                (det.get("iou", 0) >= t_iou and det.get("gt_id", 0) < CROWD_ID_T)
                for det in self.detections_by_score
            ]
        ).astype(float)

    def fp_sum_all(self, t_iou: float = 0.5):
        return np.cumsum(
            [
                (det.get("iou", 0) < t_iou and det.get("gt_id", 0) < CROWD_ID_T)
                for det in self.detections_by_score
            ]
        ).astype(float)

    def ex_sum_all(self):
        return np.cumsum(
            [det.get("gt_id", 0) > CROWD_ID_T for det in self.detections_by_score]
        ).astype(float)

    def scores_all(self):
        return [det["score"] for det in self.detections_by_score]
    def pr_curve(self, category: str, t_iou: float = 0.5, t_score: float = None):
        """Return the precision-recall curve sampled at RECALL_THRS."""
        TP = self._tp_sum(category, t_iou)
        FP = self._fp_sum(category, t_iou)
        if t_score is not None:
            scores = [-x["score"] for x in self.all_detections_by_class(category)]
            idx = np.searchsorted(scores, -t_score, "right")
            TP = TP[:idx]
            FP = FP[:idx]
        nGT = self.num_gt_class(category)
        TPR = TP / nGT
        PPV = TP / (TP + FP + np.spacing(1))
        PPVi = interpolated_PPV(PPV)
        inds = np.searchsorted(TPR, self.RECALL_THRS, side="left")
        q = np.zeros_like(self.RECALL_THRS)
        for ri, pi in enumerate(inds):
            if pi >= len(PPVi):
                break
            q[ri] = PPVi[pi]
        return q

    def average_precision(
        self, category: str, t_iou: float = 0.5, t_score: float = None
    ):
        q = self.pr_curve(category, t_iou, t_score)
        return np.mean(q)

    def mean_average_precision(
        self, t_iou: float = 0.5, t_score: float = None, rich=False
    ):
        """mAP metric."""
        if not rich:
            return np.mean(
                [self.average_precision(c, t_iou, t_score) for c in self.names]
            )
        aps = {c: self.average_precision(c, t_iou) for c in self.names}
        aps[f"mAP{t_iou}"] = np.mean(list(aps.values()))
        return aps

    def AP_score_cat(self, category: str, t_score: float = None, rich=False):
        aps = {}
        for i, t_iou in enumerate(self.IOU_THRS):
            self.match_detections(i)
            aps[t_iou] = self.average_precision(category, t_iou, t_score)
        if rich:
            return aps
        return np.mean(list(aps.values()))

    def AP_score(self, t_score: float = None, rich=False):
        aps = defaultdict(dict)
        for i, t_iou in enumerate(self.IOU_THRS):
            self.match_detections(i)
            for category in self.names:
                aps[category][t_iou] = self.average_precision(category, t_iou, t_score)
        cat_aps = {k: np.mean(list(v.values())) for k, v in aps.items()}
        ap = np.mean(list(cat_aps.values()))
        if not rich:
            return ap
        return {"AP": ap, "AP-class": cat_aps, "AP-class-iou": aps}

    def coco_AP_score(self):
        self._ensure_cocoeval()
        return np.mean(
            self.coco.eval["precision"][:, :, :, self.ALL_AREA, self.TOP_MAX_DET]
        )

    def coco_AP_cat(self, category):
        self._ensure_cocoeval()
        class_id = self.names.name_to_idx(category)
        return np.mean(
            self.coco.eval["precision"][:, :, class_id, self.ALL_AREA, self.TOP_MAX_DET]
        )

    def coco_mAP_score(self, iou_index=0):
        self._ensure_cocoeval()
        return np.mean(
            self.coco.eval["precision"][
                iou_index, :, :, self.ALL_AREA, self.TOP_MAX_DET
            ]
        )

    def coco_mAP_cat(self, category, iou_index=0):
        self._ensure_cocoeval()
        class_id = self.names.name_to_idx(category)
        return np.mean(
            self.coco.eval["precision"][
                iou_index, :, class_id, self.ALL_AREA, self.TOP_MAX_DET
            ]
        )

    def count_TP(self, t_iou=0.5, t_score=0):
        return sum(
            (
                det.get("iou", 0) >= t_iou
                and det.get("gt_id", 0) < CROWD_ID_T
                and det["score"] >= t_score
            )
            for det in self.detections
        )

    def count_FP(self, t_iou=0.5, t_score=0):
        return sum(
            (det.get("iou", 0) < t_iou and det["score"] >= t_score)
            for det in self.detections
        )

    def count_EX(self, t_iou=0.5, t_score=0):
        return sum(
            (
                det.get("iou", 0) >= t_iou
                and det.get("gt_id", 0) > CROWD_ID_T
                and det["score"] >= t_score
            )
            for det in self.detections
        )

    def summary(self, t_iou=0.5, t_score=0):
        FP = self.count_FP(t_iou, t_score)
        TP = self.count_TP(t_iou, t_score)
        PPV = TP / (TP + FP)
        TPR = TP / self._num_gt_all
        return {
            "FP": FP,
            "TP": TP,
            "PPV": PPV,
            "TPR": TPR,
            "EX": self.count_EX(t_iou, t_score),
            "F1": 2 * PPV * TPR / (PPV + TPR + np.spacing(1)),
            "T_c": t_score,
        }


def _main():
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("detection_files", nargs="+")
    parser.add_argument("--cocoeval", "-c", action="store_true")
    parser.add_argument("--gt-match", "-g", action="store_true")
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--no-cache", "-C", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    if args.debug:
        aspectize(DetectionResults, logged)

    for detection_file in args.detection_files:
        res = DetectionResults(
            detection_file,
            area_rng=None if args.cocoeval else False,
            cache=not args.no_cache,
            debug=int(args.verbose),
            gt_match=args.gt_match,
            iou_thresh=None if args.cocoeval else args.min_iou,
        )
        res.match_detections(5)
        print("mAP@.75:")
        pprint.pprint(res.mean_average_precision(0.75, rich=True))
        res.match_detections(0)
        print("mAP@.5:")
        pprint.pprint(res.mean_average_precision(rich=True))
        if args.cocoeval:
            res.finish_cocoeval()


if __name__ == "__main__":
    start = time.time()
    try:
        _main()
    finally:
        print(f"Done: {time.time()-start:.3f} s")
