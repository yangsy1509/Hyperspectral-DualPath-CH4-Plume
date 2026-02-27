#!/usr/bin/env bash
python train.py --config configs/enh/cnn_aspp.yaml
python train.py --config configs/enh/inception_unet.yaml
python train.py --config configs/enh/segformer.yaml
python train.py --config configs/rad/cnn_aspp.yaml
python train.py --config configs/rad/inception_unet.yaml
python train.py --config configs/rad/segformer.yaml
