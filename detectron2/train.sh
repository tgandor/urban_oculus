#!/bin/bash

# forwarding args; use for --resume (mainly)
python train_net.py --config-file "$1" --num-gpus $(nvidia-smi -L  | wc -l) "$@"
