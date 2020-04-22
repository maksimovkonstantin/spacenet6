#!/usr/bin/env bash
rm -rf /wdata/*
# make directories
mkdir -p /wdata/train_masks/ /wdata/folds_predict/ /wdata/pretrained_models/checkpoints/ /wdata/submits/ /wdata/segmentation_logs/

python3 /project/utils/create_masks.python
python3 /project/utils/create_folds.py

python3 /project/train/train_segmentation.py --config_path /project/configs/densenet161_gcc_fold1.py --gpu 0 & \
python3 /project/train/train_segmentation.py --config_path /project/configs/densenet161_gcc_fold2.py --gpu 1 & \
python3 /project/train/train_segmentation.py --config_path /project/configs/densenet161_gcc_fold3.py --gpu 2 & \
python3 /project/train/train_segmentation.py --config_path /project/configs/densenet161_gcc_fold4.py --gpu 3


python3 /project/train/train_segmentation.py --config_path /project/configs/dpn92_gcc_fold1.py --gpu 0 & \
python3 /project/train/train_segmentation.py --config_path /project/configs/dpn92_gcc_fold2.py --gpu 1 & \
python3 /project/train/train_segmentation.py --config_path /project/configs/dpn92_gcc_fold3.py --gpu 2 & \
python3 /project/train/train_segmentation.py --config_path /project/configs/dpn92_gcc_fold4.py --gpu 3