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
parser.add_argument('--min-score', '-t', type=float, help='confidence threshold for detections')
args = parser.parse_args()

gt = COCO(ANNOTATIONS)

detFile = args.detection_file
if detFile.endswith('.bz2'):
    detFile = json.load(bz2.open(detFile))
elif detFile.endswith('.gz'):
    detFile = json.load(gzip.open(detFile))
elif detFile.endswith('.json'):
    detFile = json.load(open(detFile))
else:
    raise ValueError(f'Unrecognized file type: {detFile}')

print(f'Loaded {len(detFile)} detections.')
if args.min_score:
    detFile = [d for d in detFile if d['score'] > args.min_score]
    print(f'Filtered with threshold {args.min_score} to {len(detFile)} detections.')

dt = gt.loadRes(detFile)

coco = COCOeval(gt, dt, iouType='bbox')
coco.params.areaRng = [[0.0, 1e9]]  # don't evalImage
coco.evaluate()
# coco.accumulate() - not needed for evalImgs...

tp = 0
fp = 0
gt = 0

for img in coco.evalImgs:
    if img is None:
        continue

    tp += (img['dtMatches'][0] > 0).sum()
    gt += len(img['gtIds']) - img['gtIgnore'].astype(int).sum()

print(f'Total objects found in {args.detection_file}: {tp:,} (of {gt:,})')

with open('evaluate_log.txt', 'a') as log:
    print(f'{args.detection_file}: {tp:,} (of {gt:,})', file=log)
