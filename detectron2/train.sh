#!/bin/bash

# forwarding args; use for --resume (mainly)
config="$1"
shift
python train_net.py --config-file "$config" --num-gpus $(nvidia-smi -L  | wc -l) "$@"
