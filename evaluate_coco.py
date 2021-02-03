import argparse
import bz2
import glob
import gzip
import json
import os

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from imports import load

ANNOTATIONS = 'datasets/coco/annotations/instances_val2017.json'
# same as:
# from detectron2.data import MetadataCatalog
# ANNOTATIONS = MetadataCatalog.get('coco_2017_val').json_file

if not os.path.exists(ANNOTATIONS):
    print('Please symlink datasets/ or unzip annotations_trainval2017.zip to datasets/coco/')
    exit()

parser = argparse.ArgumentParser()
parser.add_argument(
    'detection_file', help='path to coco_instances_results.json[.gz|.bz2|]')
parser.add_argument('--min-score', '-t', type=float,
                    help='confidence threshold for detections')
parser.add_argument('--full', '-f', action='store_true',
                    help='perform full accumulate / summarize')
args = parser.parse_args()

gt = COCO(ANNOTATIONS)

metrics = []

detFile = args.detection_file

if os.path.isdir(detFile):
    dump_dir = detFile
    detFile = glob.glob(os.path.join(
        dump_dir, "coco_instances_results.json*"))[0]
else:
    dump_dir = os.path.dirname(detFile)

results_file = glob.glob(os.path.join(
    dump_dir, "results.json*"))[0]

results = load(results_file)

detFile = load(detFile)

print(f'Loaded {len(detFile)} detections.')
if args.min_score:
    detFile = [d for d in detFile if d['score'] > args.min_score]
    print(
        f'Filtered with threshold {args.min_score} to {len(detFile)} detections.')

dt = gt.loadRes(detFile)

coco = COCOeval(gt, dt, iouType='bbox')

if not args.full:
    # don't evalImage for 'small', 'medium', 'large'
    coco.params.areaRng = [[0.0, 1e9]]

coco.evaluate()

if args.full:
    coco.accumulate()
    coco.summarize()

tp = 0
fp = 0
n_gt = 0
an_gt = 0  # alternative (no ignore)

IoU_T_IDX = 0  # first IoU threshold = 0.5

nCats = len(coco.params.catIds)
nArea = len(coco.params.areaRng)
nImgs = len(coco.params.imgIds)

assert len(coco.evalImgs) == nCats * nArea * nImgs

print(f'nArea: {nArea}, len(evalImgs): {len(coco.evalImgs)}')
for catIx in range(nCats):
    offs = catIx * (nArea * nImgs)
    for img in coco.evalImgs[offs:offs+nImgs]:
        if img is None:
            continue

        tp += (img['dtMatches'][IoU_T_IDX] > 0).sum()
        fp += (img['dtMatches'][IoU_T_IDX] == 0).sum()
        n_gt += len(img['gtIds']) - img['gtIgnore'].astype(int).sum()
        an_gt += len(img['gtIds'])

recall = tp / n_gt
precision = tp / (tp + fp)
assert tp + fp == len(detFile)
alt_recall = tp / an_gt

MAXDET_IDX = -1  # last "maxDets"
AREARNG_IDX = 0  # 'all'

print(f'Total objects found in {args.detection_file}: {tp:,} (of {n_gt:,})')
print(f'precision {precision*100:.1f} recall {recall*100:.1f}')
print(f'alt_recall {alt_recall*100:.1f}')

if args.full:
    raw_recalls = coco.eval["recall"][IoU_T_IDX, :, AREARNG_IDX, MAXDET_IDX]
    e_recall = np.mean(raw_recalls[raw_recalls > -1])
    print(f'Recall by .eval: {e_recall}')

model = results['model'].replace('_', r'\_')
ap = results['results']['bbox']['AP']
apl = results['results']['bbox']['APl']
apm = results['results']['bbox']['APm']
aps = results['results']['bbox']['APs']

metrics.append([
    model,
    ap,
    apl,
    apm,
    aps,
])

print(
    results['model'],
    {
        k: np.round(v, 1)
        for k, v in results['results']['bbox'].items() if '-' not in k
    }
)

with open('evaluate_log.txt', 'a') as log:
    print(f'{args.detection_file}: {tp:,} (of {n_gt:,}), precision {precision*100:.1f} recall {recall*100:.1f}', file=log)

print(r'Model & AP & APl & APm & APs \\')
for row in metrics:
    model, ap, apl, apm, aps = row
    print(f'{model} & {ap:.1f} & {apl:.1f} & {apm:.1f} & {aps:.1f} \\\\')
