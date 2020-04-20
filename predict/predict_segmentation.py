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
from dataset.semseg_dataset import TestSemSegDataset
from torch.utils.data import DataLoader
from fire import Fire


def main(test_images='/data/SN6_buildings/test_public/AOI_11_Rotterdam/SAR-Intensity/',
         test_predict_result='/wdata/segmentation_test_results'):
    with torch.no_grad():
        args = parse_config_args()
        config = get_config(args.config)
        model_name = config['model_name']
        weights_path = config['load_from']
        device = config['device']
        val_batch_size = config['val_batch_size']
        input_channels = config['input_channels']

        original_size = config['original_size']
        cropper = albu.Compose([albu.CenterCrop(original_size[0], original_size[1], p=1.0)])
        n_classes = config['n_classes']
        preprocessing_fn = config['preprocessing_fn']
        valid_augs = config['valid_augs']
        limit_files = config['limit_files']
        num_workers = config['num_workers']

        test_dataset = TestSemSegDataset(images_dir=test_images,
                                         preprocessing=preprocessing_fn,
                                         augmentation=valid_augs,
                                         limit_files=limit_files)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=val_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

        model = make_model(
            model_name=model_name,
            weights=None,
            n_classes=n_classes,
            input_channels=input_channels).to(device)

        model.load_state_dict(torch.load(weights_path)['model_state_dict'])

        model.eval()
        model = tta.TTAWrapper(model, flip_image2mask)

        file_names = sorted(test_dataset.ids)

        for batch_i, test_batch in enumerate(tqdm(test_loader)):
            runner_out = model(test_batch.cuda())
            image_pred = runner_out

            image_pred = image_pred.cpu().detach().numpy()
            names = file_names[batch_i * val_batch_size:(batch_i + 1) * val_batch_size]
            for i in range(len(names)):
                file_name = os.path.join(test_predict_result, names[i])
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
    main()


