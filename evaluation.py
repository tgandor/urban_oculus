
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
    path = MetadataCatalog.get('coco_2017_val').json_file
    if not os.path.exists(path):
        print(f'Missing {path}')
        print('Please symlink datasets/ or unzip annotations_trainval2017.zip to datasets/coco/')
        exit()

    return COCO(path)


def _load_detections(file_or_dir):
    if os.path.isdir(file_or_dir):
        dump_dir = file_or_dir
        det_filename = glob.glob(os.path.join(
            dump_dir, "coco_instances_results.json*"))[0]
    else:
        dump_dir = os.path.dirname(file_or_dir)
        det_filename = file_or_dir

    detections = load(det_filename)

    return detections, det_filename


class DetectionResults:
    def __init__(self, det_file_or_dir, dataset='coco_2017_val') -> None:
        self.detections, self.det_file = _load_detections(det_file_or_dir)
        self.gt = _load_gt(dataset)
        self.names = Names(MetadataCatalog.get(dataset))
        self.dt = self.gt.loadRes(self.detections)
        self.coco = COCOeval(self.gt, self.dt, iouType='bbox')
        self._evaluate()
        self._enrich_detections()

    def _evaluate(self, use_cats=True):
        # don't evalImage for 'small', 'medium', 'large'
        self.coco.params.areaRng = [[0.0, 1e9]]
        # don't segregate objects by category
        self.coco.params.useCats = use_cats
        # usa a single IoU threshold
        self.coco.params.iouThrs = [0.5]

        self.coco.evaluate()

        nCats = len(self.coco.params.catIds) if use_cats else 1
        nArea = len(self.coco.params.areaRng)
        nImgs = len(self.coco.params.imgIds)

        print('nCats:', nCats, len(self.coco.evalImgs), nCats * nArea * nImgs)
        assert len(self.coco.evalImgs) == nCats * nArea * nImgs

        for catIx in range(nCats):
            catId = self.coco.params.catIds[catIx] if use_cats else -1
            offs = catIx * (nArea * nImgs)
            for imgIx, img in enumerate(self.coco.evalImgs[offs:offs+nImgs]):
                if img is None:
                    continue
                imgId = self.coco.params.imgIds[imgIx]
                ious = self.coco.ious[imgId, catId]
                dt_id2dind = {dt_id: dind for dind, dt_id in enumerate(img["dtIds"])}
                for gind, fdt_id in enumerate(img["gtMatches"][0]):
                    dt_id = int(fdt_id)
                    if not dt_id:
                        continue
                    self.detections[dt_id-1]["true_positive"] = True
                    dind = dt_id2dind[dt_id]
                    self.detections[dt_id-1]["iou"] = ious[dind, gind]

    def _enrich_detections(self):
        for d in self.detections:
            d: dict
            del d['segmentation']
            del d['id']
            del d['iscrowd']  # for now?
            d['score'] = round(d['score'], 5)
            d.setdefault('true_positive', False)
            if 'iou' in d:
                d['iou'] = round(d['iou'], 3)
            d['bbox'] = [round(x, 1) for x in d['bbox']]
            d['area'] = round(d['area'], 1)
            # last key:
            d['category'] = self.names.get(d['category_id'])
            del d['category_id']


def _main():
    res = DetectionResults(sys.argv[1])
    for d in sorted(res.detections, key=itemgetter("score"), reverse=True):
        print(d)


if __name__ == '__main__':
    start = time.time()
    try:
        _main()
    finally:
        print(f'Done: {time.time()-start:.3f} s', file=sys.stderr)
