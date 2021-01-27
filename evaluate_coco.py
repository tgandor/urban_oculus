import argparse
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANNOTATIONS = 'datasets/coco/annotations/instances_val2017.json'

if not os.path.exists(ANNOTATIONS):
    print('Please symlink datasets/ or unzip annotations_trainval2017.zip to datasets/coco/')
    exit()

parser = argparse.ArgumentParser()
parser.add_argument('detection_file', help='path to coco_instances_results.json')
args = parser.parse_args()

gt = COCO(ANNOTATIONS)
dt = gt.loadRes(args.detection_file)

coco = COCOeval(gt, dt, iouType='bbox')
coco.evaluate()
coco.accumulate()

import code; code.interact(local=locals())
