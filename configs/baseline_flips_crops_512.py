import albumentations as albu
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from dataset.semseg_dataset import SemSegDataset, TestSemSegDataset

data_type = 'SAR-Intensity'
test_images = '/data/SN6_buildings/test_public/AOI_11_Rotterdam/SAR-Intensity/'
train_images = '/data/SN6_buildings/train/AOI_11_Rotterdam/'
masks_data_path = '/wdata/train_masks'
logs_path = '/wdata/segmentation_logs/'
folds_file = '/wdata/folds.csv'
load_from = '/wdata/segmentation_logs/512_1_unet_resnet34/checkpoints/best.pth'
validation_predict_result = '/wdata/segmentation_validation_results'
test_predict_result = '/wdata/segmentation_test_results'
submit_path = '/wdata/submits/baseline.csv'
optical_pretrain = '/wdata/segmentation_logs/baseline_psrgb_1_unet_resnet34/checkpoints/best.pth'

main_metric = 'dice'
minimize_metric = False
scheduler_mode = 'max'
device = 'cuda'
fold_number = 1
n_classes = 2
input_channels = 4
crop_size = (512, 512)
val_size = (928, 928)
original_size = (900, 900)

batch_size = 16
num_workers = 8
val_batch_size = 8

shuffle = True
lr = 1e-4
momentum = 0.0
decay = 0.0
loss = 'focal_dice'
optimizer = 'radam'
fp16 = False

alias = '512_'
model_name = 'unet_resnet34'
scheduler = 'reduce_on_plateau'
patience = 10

early_stopping = 30
min_delta = 0.005

alpha = 0.5
augs_p = 0.5
min_lr = 1e-6
thershold = 0.005
best_models_count = 5

epochs = 300
weights = 'imagenet'
limit_files = None # for debug


# preprocessing_fn = smp.encoders.get_preprocessing_fn('_'.join(model_name.split('_')[1:]), weights)
# preprocessing_fn = lambda x: x / 255.0

preprocessing_fn = None

train_augs = albu.Compose([albu.OneOf([albu.RandomCrop(crop_size[0], crop_size[1], p=1.0),
                                       albu.RandomSizedCrop((int(crop_size[0] * 0.9), int(crop_size[1] * 1.1)),
                                                            crop_size[0], crop_size[1], p=1.0)
                                       ], p=1.0),
                           albu.OneOf([albu.HorizontalFlip(p=augs_p),
                                       albu.VerticalFlip(p=augs_p)], p=augs_p)
                           ], p=augs_p)

valid_augs = albu.Compose([albu.PadIfNeeded(min_height=val_size[0], min_width=val_size[1], p=1.0)])
# valid_augs = None
train_dataset = SemSegDataset(images_dir=train_images,
                              data_type=data_type,
                              masks_dir=masks_data_path,
                              mode='train',
                              folds_file=folds_file,
                              fold_number=fold_number,
                              augmentation=train_augs,
                              preprocessing=preprocessing_fn,
                              limit_files=limit_files)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers)

valid_dataset = SemSegDataset(images_dir=train_images,
                              data_type=data_type,
                              mode='valid',
                              folds_file=folds_file,
                              fold_number=fold_number,
                              augmentation=valid_augs,
                              preprocessing=preprocessing_fn,
                              limit_files=limit_files)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=val_batch_size,
                          shuffle=False,
                          num_workers=num_workers)

test_dataset = TestSemSegDataset(images_dir=test_images,
                                 preprocessing=preprocessing_fn,
                                 augmentation=valid_augs,
                                 limit_files=limit_files)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=val_batch_size,
                         shuffle=False,
                         num_workers=num_workers)


loaders = {'train': train_loader, 'valid': valid_loader}


if __name__  == '__main__':
    pass
