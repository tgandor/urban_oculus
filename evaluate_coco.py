import argparse
import bz2
import gzip
import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANNOTATIONS = 'datasets/coco/annotations/instances_val2017.json'

if not os.path.exists(ANNOTATIONS):
    print('Please symlink datasets/ or unzip annotations_trainval2017.zip to datasets/coco/')
    exit()

parser = argparse.ArgumentParser()
parser.add_argument('detection_file', help='path to coco_instances_results.json[.gz|.bz2|]')
args = parser.parse_args()

gt = COCO(ANNOTATIONS)

detFile = args.detection_file
if detFile.endswith('.bz2'):
    detFile = json.load(bz2.open(detFile))
elif detFile.endswith('.gz'):
    detFile = json.load(gzip.open(detFile))

dt = gt.loadRes(detFile)

coco = COCOeval(gt, dt, iouType='bbox')
coco.evaluate()
# coco.accumulate() - not needed for evalImgs...

tp = 0

for img in coco.evalImgs:
    if img is None:
        continue
    # print(img)
    # break
    tp += (img['dtMatches'][0] > 0).sum()

print(f'Total objects found in {args.detection_file}: {tp:,}')

with open('evaluate_log.txt', 'a') as log:
    print(f'{args.detection_file}: {tp:,}', file=log)
