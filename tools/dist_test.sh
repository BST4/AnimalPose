#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT ${@:4}
