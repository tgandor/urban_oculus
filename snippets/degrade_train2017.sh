#!/bin/bash

time find $HOME/datasets/coco/train2017/ -type f | parallel mogrify -quality $1
