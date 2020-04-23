#!/usr/bin/env bash
rm -rf /wdata/*
# make directories
mkdir -p /wdata/train_masks/ /wdata/folds_predict/ /wdata/pretrained_models/checkpoints/ /wdata/submits/ /wdata/segmentation_logs/

wget http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth /wdata/pretrained_models/checkpoints/
wget http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-fda993c95.pth /wdata/pretrained_models/checkpoints/
wget http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth /wdata/pretrained_models/checkpoints/
wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth /wdata/pretrained_models/checkpoints/

python3 /project/utils/create_masks.py
python3 /project/utils/create_folds.py

python3 /project/train/train_segmentation.py --config_path /project/configs/densenet161_gcc_fold1.py --gpu '"0"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/densenet161_gcc_fold2.py --gpu '"1"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/densenet161_gcc_fold3.py --gpu '"2"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/densenet161_gcc_fold4.py --gpu '"3"'


python3 /project/train/train_segmentation.py --config_path /project/configs/dpn92_gcc_fold1.py --gpu '"0"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/dpn92_gcc_fold2.py --gpu '"1"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/dpn92_gcc_fold3.py --gpu '"2"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/dpn92_gcc_fold4.py --gpu '"3"'

python3 /project/train/train_segmentation.py --config_path /project/configs/effnetb7_gcc_fold1.py --gpu '"0"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/effnetb7_gcc_fold2.py --gpu '"1"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/effnetb7_gcc_fold3.py --gpu '"2"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/effnetb7_gcc_fold4.py --gpu '"3"'

python3 /project/train/train_segmentation.py --config_path /project/configs/senet154_gcc_fold1.py --gpu '"0"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/senet154_gcc_fold2.py --gpu '"1"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/senet154_gcc_fold3.py --gpu '"2"' & \
python3 /project/train/train_segmentation.py --config_path /project/configs/senet154_gcc_fold4.py --gpu '"3"'