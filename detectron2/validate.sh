#!/bin/bash

# Cheat sheet:
# [ (i, len(range(0, 100, i)), [101 - x for x in range(0, 100, i)]) for i in range(6, 20)]
# [(6, 17, [101, 95, 89, 83, 77, 71, 65, 59, 53, 47, 41, 35, 29, 23, 17, 11, 5]),
#  (7, 15, [101, 94, 87, 80, 73, 66, 59, 52, 45, 38, 31, 24, 17, 10, 3]),
#  (8, 13, [101, 93, 85, 77, 69, 61, 53, 45, 37, 29, 21, 13, 5]),
#  (9, 12, [101, 92, 83, 74, 65, 56, 47, 38, 29, 20, 11, 2]),
#  (10, 10, [101, 91, 81, 71, 61, 51, 41, 31, 21, 11]),
#  (11, 10, [101, 90, 79, 68, 57, 46, 35, 24, 13, 2]),
#  (12, 9, [101, 89, 77, 65, 53, 41, 29, 17, 5]),
#  (13, 8, [101, 88, 75, 62, 49, 36, 23, 10]),
#  (14, 8, [101, 87, 73, 59, 45, 31, 17, 3]),
#  (15, 7, [101, 86, 71, 56, 41, 26, 11]),
#  (16, 7, [101, 85, 69, 53, 37, 21, 5]),
#  (17, 6, [101, 84, 67, 50, 33, 16]),
#  (18, 6, [101, 83, 65, 47, 29, 11]),
#  (19, 6, [101, 82, 63, 44, 25, 6])]

for dir in "$@" ; do
    base=`basename $dir /`
    echo `date` $base
    time python ../custom_validate_coco.py $base $base/config.yaml $base/model_final.pth -m 5 -M 101 -S 7
    echo `date` "$base done."
done
