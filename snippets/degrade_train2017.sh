#!/bin/bash

if [ "$1" == "" ] ; then
    echo "Usage: $0 <QUALITY>"
fi

if [ -f "$HOME/train2017.zip" ] ; then
    mkdir -p "$HOME/datasets/coco/"
    unzip -o "$HOME/train2017.zip" -d "$HOME/datasets/coco/"
fi

time find $HOME/datasets/coco/train2017/ -type f | parallel mogrify -quality $1

echo "Testing..."
pushd $HOME/datasets/coco/train2017/
ls | head | xargs ~/meats/dates/jpeg_quality.sh
