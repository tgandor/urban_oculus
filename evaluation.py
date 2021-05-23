
import glob
import os
import sys

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


    def rich_detections(self):
        for d in self.detections:
            d['category'] = self.names.get(d['category_id'])
            yield d


if __name__ == '__main__':
    res = DetectionResults(sys.argv[1])
    for d in res.rich_detections():
        print(d)
        break
