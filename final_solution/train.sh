#!/usr/bin/env bash
ARG1=${1:-/data/SN6_buildings/test_public/AOI_11_Rotterdam/}
rm -rf /wdata/*
# make directories
mkdir -p /wdata/train_masks/  /wdata/pretrained_models/checkpoints/ /wdata/segmentation_logs/ /wdata/final_models/

# create masks and folds
python3 /project/utils/create_masks.py --data_root_path $ARG1
python3 /project/utils/create_folds.py --images_path $ARG1

# load pretrained
wget http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth /wdata/pretrained_models/checkpoints/

python3 /project/train/train_segmentation.py --data_path $ARG1 --config_path /project/configs/senet154_gcc_fold1.py --gpu '"0"' \
& python3 /project/train/train_segmentation.py --data_path $ARG1 --config_path /project/configs/senet154_gcc_fold2.py --gpu '"1"' \
& python3 /project/train/train_segmentation.py --data_path $ARG1 --config_path /project/configs/senet154_gcc_fold3.py --gpu '"2"' \
& python3 /project/train/train_segmentation.py --data_path $ARG1 --config_path /project/configs/senet154_gcc_fold4.py --gpu '"3"' & wait

python3 /project/train/train_segmentation.py --data_path $ARG1 --config_path /project/configs/senet154_gcc_fold5.py --gpu '"0"' \
& python3 /project/train/train_segmentation.py --data_path $ARG1 --config_path /project/configs/senet154_gcc_fold6.py --gpu '"1"' \
& python3 /project/train/train_segmentation.py --data_path $ARG1 --config_path /project/configs/senet154_gcc_fold7.py --gpu '"2"' \
& python3 /project/train/train_segmentation.py --data_path $ARG1 --config_path /project/configs/senet154_gcc_fold8.py --gpu '"3"' & wait
