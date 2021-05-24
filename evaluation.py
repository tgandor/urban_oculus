import glob
from operator import itemgetter
import os
import sys
import time

from detectron2.data import (  # noqa
    DatasetCatalog,
    MetadataCatalog,
)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from imports import load, Names


def _load_gt(dataset):
    path = MetadataCatalog.get("coco_2017_val").json_file
    if not os.path.exists(path):
        print(f"Missing {path}")
        print(
            "Please symlink datasets/ or unzip annotations_trainval2017.zip to datasets/coco/"
        )
        exit()

    return COCO(path)


def _load_detections(file_or_dir):
    if os.path.isdir(file_or_dir):
        dump_dir = file_or_dir
        det_filename = glob.glob(
            os.path.join(dump_dir, "coco_instances_results.json*")
        )[0]
    else:
        dump_dir = os.path.dirname(file_or_dir)
        det_filename = file_or_dir

    detections = load(det_filename)

    return detections, det_filename


class DetectionResults:
    def __init__(
        self,
        det_file_or_dir,
        *,
        dataset="coco_2017_val",
        use_cats=True,
        area_rng=False,
        iou_thresh=(0.5,),
    ):
        self.detections, self.det_file = _load_detections(det_file_or_dir)
        self.gt = _load_gt(dataset)
        self.names = Names(MetadataCatalog.get(dataset))
        self.dt = self.gt.loadRes(self.detections)
        self.coco = COCOeval(self.gt, self.dt, iouType="bbox")
        # COCOeval params
        self.use_cats = use_cats
        self.area_rng = area_rng
        self.iou_thresh = iou_thresh
        # process
        self._evaluate()
        self._enrich_detections()

    def _evaluate(self):
        # don't evalImage for 'small', 'medium', 'large'
        if self.area_rng is False:
            self.coco.params.areaRng = [[0.0, 1e9]]
        elif self.area_rng is not None:
            self.coco.params.areaRng = self.area_rng
        # don't segregate objects by category
        self.coco.params.useCats = self.use_cats
        # usa a single IoU threshold
        if self.iou_thresh is not None:
            self.coco.params.iouThrs = self.iou_thresh

        self.coco.evaluate()

        nCats = len(self.coco.params.catIds) if self.use_cats else 1
        nArea = len(self.coco.params.areaRng)
        nImgs = len(self.coco.params.imgIds)

        print("nCats:", nCats, len(self.coco.evalImgs), nCats * nArea * nImgs)
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
                    self.detections[dt_id - 1]["true_positive"] = True
                    dind = dt_id2dind[dt_id]
                    self.detections[dt_id - 1]["iou"] = ious[dind, gind]

    def _enrich_detections(self):
        for d in self.detections:
            d: dict
            del d["segmentation"]
            del d["id"]
            del d["iscrowd"]  # for now?
            d["score"] = round(d["score"], 5)
            d.setdefault("true_positive", False)
            if "iou" in d:
                d["iou"] = round(d["iou"], 3)
            d["bbox"] = [round(x, 1) for x in d["bbox"]]
            d["area"] = round(d["area"], 1)
            # last key:
            d["category"] = self.names.get(d["category_id"])
            del d["category_id"]

    @property
    def num_gt_all(self):
        return sum(g["ignore"] == 0 for v in self.coco._gts.values() for g in v)

    @property
    def detections_by_score(self):
        return sorted(self.detections, key=itemgetter("score"), reverse=True)

    def detections_by_class(self, name: str):
        return [d for d in self.detections_by_score if d["category"] == name]

    def num_gt_class(self, name):
        cls_id = self.names.name_to_id(name)
        return sum(
            g["ignore"] == 0
            for (_, cat_id), v in self.coco._gts.items()
            for g in v
            if cat_id == cls_id
        )


def _main():
    res = DetectionResults(sys.argv[1])
    for d in res.detections_by_score:
        print(d)


if __name__ == "__main__":
    start = time.time()
    try:
        _main()
    finally:
        print(f"Done: {time.time()-start:.3f} s", file=sys.stderr)
