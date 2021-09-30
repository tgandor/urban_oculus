#!/bin/bash

time parallel python evaluate_coco.py ::: ~/reval_50/*/*
