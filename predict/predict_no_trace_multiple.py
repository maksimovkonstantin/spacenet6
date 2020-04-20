import torch
import albumentations as albu
import os
import numpy as np
import rasterio
from tqdm import tqdm
from tta import flip_image2mask
from pytorch_toolbelt.inference import tta
from torch.utils.data import DataLoader
from dataset.semseg_dataset import TestSemSegDataset
from fire import Fire
from models.model_factory import make_model

def main(test_images='/data/SN6_buildings/test_public/AOI_11_Rotterdam/SAR-Intensity/',
         test_predict_result='/wdata/segmentation_test_results'):
    with torch.no_grad():
        batch_size = 8
        workers = 8
        test_augs = albu.Compose([albu.PadIfNeeded(min_height=928, min_width=928, p=1.0)])
        cropper = albu.Compose([albu.CenterCrop(900, 900, p=1.0)])
        test_dataset = TestSemSegDataset(images_dir=test_images,
                                         preprocessing=None,
                                         augmentation=test_augs,
                                         limit_files=None)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=workers)

        paths = ['/wdata/segmentation_logs/newsteps_5folds_steps_adam_gcc_1_unet_densenet161/checkpoints/best.pth',
                 '/wdata/segmentation_logs/newsteps_5folds_steps_adam_gcc_1_unet_dpn92/checkpoints/best.pth',
                 '/wdata/segmentation_logs/newsteps_5folds_steps_adam_gcc_1_unet_efficientnet-b7/checkpoints/best.pth',

                 '/wdata/segmentation_logs/newsteps_5folds_steps_adam_gcc_2_unet_densenet161/checkpoints/best.pth',
                 '/wdata/segmentation_logs/newsteps_5folds_steps_adam_gcc_2_unet_dpn92/checkpoints/best.pth',
                 '/wdata/segmentation_logs/newsteps_5folds_steps_adam_gcc_2_unet_efficientnet-b7/checkpoints/best.pth'
                 #'/wdata/traced_models/adam_gcc_1_unet_densenet161.pth',
                 #'/wdata/traced_models/adam_gcc_1_unet_dpn92.pth'
                 ]

        models = []
        device = 'cuda'
        n_classes = 2
        input_channels = 4
        for weights_path in paths:
            print('loaded {}'.format (weights_path))
            model_name  = '_'.join(weights_path.split('/')[-3].split('_')[-2:])
            model = make_model(
                model_name=model_name,
                weights=None,
                n_classes=n_classes,
                input_channels=input_channels).to(device)
            # model = torch.jit.load(weights_path)
            model.eval()
            model = tta.TTAWrapper(model, flip_image2mask)
            models.append(model)

        file_names = sorted(test_dataset.ids)
        for batch_i, test_batch in enumerate(tqdm(test_loader)):
            for model_i, model in enumerate(models):
                runner_out = model(test_batch.cuda())
                # runner_out = torch.sigmoid(runner_out)
                if model_i == 0:
                    image_pred = runner_out.cpu().detach().numpy()
                else:
                    image_pred += runner_out.cpu().detach().numpy()
                # image_pred = runner_out
            # image_pred = model(test_batch.cuda())
            # image_pred = image_pred.cpu().detach().numpy()
            image_pred = image_pred / len(models)
            names = file_names[batch_i*batch_size:(batch_i+1)*batch_size]
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
    Fire(main)
