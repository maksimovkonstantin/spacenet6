python3 predict_segmentation_png.py --config_path ../configs/effnetb7_gcc_fold1.py --batch_size 8  --gpu '"0"' \
& python3 predict_segmentation_png.py --config_path ../configs/effnetb7_gcc_fold2.py --batch_size 8 --gpu '"1"' & wait

python3 predict_segmentation_png.py --config_path ../configs/effnetb7_gcc_fold3.py --batch_size 8  --gpu '"0"' \
& python3 predict_segmentation_png.py --config_path ../configs/effnetb7_gcc_fold4.py --batch_size 8 --gpu '"1"' & wait

python3 predict_segmentation_png.py --config_path ../configs/senet154_gcc_fold1.py --batch_size 8  --gpu '"0"' \
& python3 predict_segmentation_png.py --config_path ../configs/senet154_gcc_fold2.py --batch_size 8 --gpu '"1"' & wait

python3 predict_segmentation_png.py --config_path ../configs/senet154_gcc_fold3.py --batch_size 8  --gpu '"0"' \
& python3 predict_segmentation_png.py --config_path ../configs/senet154_gcc_fold4.py --batch_size 8 --gpu '"1"' & wait