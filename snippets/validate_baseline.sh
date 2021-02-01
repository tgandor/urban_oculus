#!/bin/bash

for model in R101 R101_C4 R101_DC5 R101_FPN R50 R50_C4 R50_DC5 R50_FPN X101 ; do
    python validate_coco.py --minQ 101 --maxQ 101 $model
done

mkdir baseline_dev_0
mv evaluator_dump_*_101 baseline_dev_0/

for model in R101 R101_C4 R101_DC5 R101_FPN R50 R50_C4 R50_DC5 R50_FPN X101 ; do
    python validate_coco.py -d 1 --minQ 101 --maxQ 101 $model
done

mkdir baseline_dev_1
mv evaluator_dump_*_101 baseline_dev_1/

# real    344m46,761s
# user    372m5,682s
# sys     4m1,770s
