python3 train_segmentation.py --config_path ../configs/senet154_gcc_fold1.py --gpu '"0"' \
& python3 train_segmentation.py --config_path ../configs/senet154_gcc_fold2.py --gpu '"1"' \
& python3 train_segmentation.py --config_path ../configs/senet154_gcc_fold3.py --gpu '"2"' \
& python3 train_segmentation.py --config_path ../configs/senet154_gcc_fold4.py --gpu '"3"' & wait

python3 train_segmentation.py --config_path ../configs/senet154_gcc_fold5.py --gpu '"0"' \
& python3 train_segmentation.py --config_path ../configs/senet154_gcc_fold6.py --gpu '"1"' \
& python3 train_segmentation.py --config_path ../configs/senet154_gcc_fold7.py --gpu '"2"' \
& python3 train_segmentation.py --config_path ../configs/senet154_gcc_fold8.py --gpu '"3"' & wait