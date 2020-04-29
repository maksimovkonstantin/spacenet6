#!/usr/bin/env bash
ARG1=${1:-/data/SN6_buildings/test_public/AOI_11_Rotterdam/}
ARG2=${2:-/wdata/solution.csv}

mkdir -p /wdata/segmentation_logs/ /wdata/folds_predictss/

if [ "$(ls -A /wdata/segmentation_logs/)" ]; then
     echo "trained weights available"
else
    echo "loading pretrained weights"
    mkdir -p /wdata/segmentation_logs/8_3_reduce_1_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_2_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_3_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_4_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_5_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_6_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_7_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_8_unet_senet154/checkpoints/
    gdown https://drive.google.com/uc?id=1XnUqlwggvQe_SbmbxNvPbbhVPaYvln7Q -O /wdata/segmentation_logs/8_3_reduce_1_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1XnUqlwggvQe_SbmbxNvPbbhVPaYvln7Q -O /wdata/segmentation_logs/8_3_reduce_2_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1XnUqlwggvQe_SbmbxNvPbbhVPaYvln7Q -O /wdata/segmentation_logs/8_3_reduce_3_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1XnUqlwggvQe_SbmbxNvPbbhVPaYvln7Q -O /wdata/segmentation_logs/8_3_reduce_4_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1XnUqlwggvQe_SbmbxNvPbbhVPaYvln7Q -O /wdata/segmentation_logs/8_3_reduce_5_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1XnUqlwggvQe_SbmbxNvPbbhVPaYvln7Q -O /wdata/segmentation_logs/8_3_reduce_6_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1XnUqlwggvQe_SbmbxNvPbbhVPaYvln7Q -O /wdata/segmentation_logs/8_3_reduce_7_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1XnUqlwggvQe_SbmbxNvPbbhVPaYvln7Q -O /wdata/segmentation_logs/8_3_reduce_8_unet_senet154/checkpoints/best.pth

fi


python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold1.py --gpu '"0"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold2.py --gpu '"1"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold3.py --gpu '"2"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold4.py --gpu '"3"' --test_images $ARG1 --workers 16 --batch_size 16 & wait

python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold5.py --gpu '"0"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold6.py --gpu '"1"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold7.py --gpu '"2"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold8.py --gpu '"3"' --test_images $ARG1 --workers 16 --batch_size 16 & wait

python3 /project/predict/submit.py --submit_path $ARG2
