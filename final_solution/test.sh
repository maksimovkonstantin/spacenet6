#!/usr/bin/env bash
ARG1=${1:-/data/SN6_buildings/test_public/AOI_11_Rotterdam/}
ARG2=${1:-/wdata/solution.csv}

mkdir -p /wdata/segmentation_logs/

python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/densenet161_gcc_fold1.py --gpu '"1"' --test_images $ARG1 --workers 8 --batch_size 8 &
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/densenet161_gcc_fold2.py --gpu '"1"' --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/densenet161_gcc_fold3.py --gpu '"1"' --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/densenet161_gcc_fold4.py --gpu '"1"' --test_images $ARG1 --workers 8 --batch_size 8

python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/dpn92_gcc_fold1.py --gpu 0 --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/dpn92_gcc_fold2.py --gpu 1 --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/dpn92_gcc_fold3.py --gpu 2 --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/dpn92_gcc_fold4.py --gpu 3 --test_images $ARG1 --workers 8 --batch_size 8

python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/effnetb7_gcc_fold1.py --gpu 0 --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/effnetb7_gcc_fold2.py --gpu 1 --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/effnetb7_gcc_fold3.py --gpu 2 --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/effnetb7_gcc_fold4.py --gpu 3 --test_images $ARG1 --workers 8 --batch_size 8

python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/senet154_gcc_fold1.py --gpu 0 --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/senet154_gcc_fold2.py --gpu 1 --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/senet154_gcc_fold3.py --gpu 2 --test_images $ARG1 --workers 8 --batch_size 8
python3 /project/predict/predict_segmentation_png.py --config_path /project/configs/senet154_gcc_fold4.py --gpu 3 --test_images $ARG1 --workers 8 --batch_size 8