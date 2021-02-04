#!/bin/bash

for model in R101 R101_C4 R101_DC5 R101_FPN R50 R50_C4 R50_DC5 R50_FPN X101 ; do
    python validate_coco.py --min-score 0.05 --minQ 101 --maxQ 101 $model
done

mkdir baseline_05
mv evaluator_dump_*_101 baseline_05

for model in R101 R101_C4 R101_DC5 R101_FPN R50 R50_C4 R50_DC5 R50_FPN X101 ; do
    python validate_coco.py --min-score 0.5 --minQ 101 --maxQ 101 $model
done

mkdir baseline_50
mv evaluator_dump_*_101 baseline_50

# Prev:
# real    344m46,761s
# user    372m5,682s
# sys     4m1,770s
