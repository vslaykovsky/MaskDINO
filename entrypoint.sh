#!/bin/sh

cd maskdino/modeling/pixel_decoder/ops/ && sh make.sh
cd /app && python train_net.py --num-gpus 1 --config-file configs/coco/instance-segmentation/hubmap.yaml