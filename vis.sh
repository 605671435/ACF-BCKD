#!/usr/bin/env bash

CONFIG=$1
CKPT=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/visualizations/vis_gt.py \
  $CONFIG \
  $CKPT \
  --save
