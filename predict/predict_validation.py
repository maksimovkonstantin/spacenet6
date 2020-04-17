import torch
from tqdm import tqdm
import os
import numpy as np
from utils.helpers import get_config, parse_config_args
from models.model_factory import make_model
import albumentations as albu
import rasterio
from pytorch_toolbelt.inference import tta
from tta import flip_image2mask
from dataset.semseg_dataset import SemSegDataset
from torch.utils.data import DataLoader
from fire import Fire


def main(test_predict_result='/wdata/segmentation_validation_results'):
    with torch.no_grad():
        args = parse_config_args()
        config = get_config(args.config)
        model_name = config['model_name']
        weights_path = config['load_from']
        device = config['device']
        input_channels = config['input_channels']
        n_classes = config['n_classes']
        original_size = config['original_size']

        train_images = config['train_images']
        data_type = config['data_type']
        folds_file = config['folds_file']
        fold_number = config['fold_number']
        preprocessing_fn = config['preprocessing_fn']
        limit_files = config['limit_files']
        num_workers = config['num_workers']
        valid_augs = config['valid_augs']
        val_batch_size = config['val_batch_size']

        cropper = albu.Compose([albu.CenterCrop(original_size[0], original_size[1], p=1.0)])
        model = make_model(
            model_name=model_name,
            weights=None,
            n_classes=n_classes,
            input_channels=input_channels).to(device)

        model.load_state_dict(torch.load(weights_path)['model_state_dict'])
        model.eval()
        model = tta.TTAWrapper(model, flip_image2mask)

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

        file_names = sorted(config['valid_dataset'].ids)

        for batch_i, test_batch in enumerate(tqdm(valid_loader)):
            runner_out = model(test_batch[0].cuda())
            image_pred = torch.sigmoid(runner_out)
            image_pred = image_pred.cpu().detach().numpy()
            names = file_names[batch_i * val_batch_size:(batch_i + 1) * val_batch_size]
            for i in range(len(names)):
                file_name = os.path.join(test_predict_result, names[i] + '.tif')
                data = image_pred[i, ...]
                data = np.moveaxis(data, 0, -1)
                sample = cropper(image=data)

                data = sample['image']
                data = np.moveaxis(data, -1, 0)
                c, h, w = data.shape
                with rasterio.open(file_name, "w", dtype=rasterio.float32, driver='GTiff',
                                   width=h, height=h, count=c) as dest:
                    dest.write(data)

if __name__ == '__main__':
   Fire(main)

