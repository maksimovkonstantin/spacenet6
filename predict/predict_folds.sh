python3 predict_segmentation_png.py --config_path ../../configs/densenet161_gcc_fold1.py --batch_size 2 --gpu 0 \
& python3 predict_segmentation_png.py --config_path ../../configs/dpn92_gcc_fold1.py --batch_size 2 --gpu 1

python3 predict_segmentation_png.py --config_path ../../configs/densenet161_gcc_fold2.py --batch_size 2 --gpu 0 \
& python3 predict_segmentation_png.py --config_path ../../configs/dpn92_gcc_fold2.py --batch_size 2 --gpu 1

python3 predict_segmentation_png.py --config_path ../../configs/densenet161_gcc_fold3.py --batch_size 2 --gpu 0 \
& python3 predict_segmentation_png.py --config_path ../../configs/dpn92_gcc_fold3.py --batch_size 2 --gpu 1
