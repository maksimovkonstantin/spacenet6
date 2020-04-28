import torch
import albumentations as albu
import os
import numpy as np
import rasterio
from tqdm import tqdm
from tta import flip_image2mask
from pytorch_toolbelt.inference import tta
from torch.utils.data import DataLoader
from dataset.semseg_dataset import SemSegDataset
from fire import Fire
from models.model_factory import make_model

def main(test_predict_result='/wdata/segmentation_validation_results'):
    with torch.no_grad():
        batch_size = 8
        workers = 8
        test_augs = albu.Compose([albu.PadIfNeeded(min_height=928, min_width=928, p=1.0)])
        cropper = albu.Compose([albu.CenterCrop(900, 900, p=1.0)])
        valid_dataset = SemSegDataset(images_dir='/data/SN6_buildings/train/AOI_11_Rotterdam/',
                                      data_type='SAR-Intensity',
                                      mode='valid',
                                      n_classes=3,
                                      folds_file='/wdata/folds.csv',
                                      fold_number=1,
                                      augmentation=test_augs,
                                      preprocessing=None,
                                      limit_files=None)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=workers)

        paths = ['/wdata/segmentation_logs/3_reduce_1_unet_densenet161/checkpoints/best.pth',
                 '/wdata/segmentation_logs/3_reduce_1_unet_dpn92/checkpoints/best.pth'
                 #'/wdata/segmentation_logs/reduce_1_unet_efficientnet-b7/checkpoints/best.pth',
                 #'/wdata/segmentation_logs/reduce_1_unet_senet154/checkpoints/best.pth'
                 #'/wdata/segmentation_logs/newsteps_5folds_steps_adam_gcc_2_unet_densenet161/checkpoints/best.pth',
                 #'/wdata/segmentation_logs/newsteps_5folds_steps_adam_gcc_2_unet_dpn92/checkpoints/best.pth',
                 #'/wdata/segmentation_logs/newsteps_5folds_steps_adam_gcc_2_unet_efficientnet-b7/checkpoints/best.pth'
                 #'/wdata/traced_models/adam_gcc_1_unet_densenet161.pth',
                 #'/wdata/traced_models/adam_gcc_1_unet_dpn92.pth'
                 ]

        models = []
        device = 'cuda'
        n_classes = 3
        input_channels = 4
        for weights_path in paths:

            model_name  = '_'.join(weights_path.split('/')[-3].split('_')[-2:])
            model = make_model(
                model_name=model_name,
                weights=None,
                n_classes=n_classes,
                input_channels=input_channels).to(device)
            model.load_state_dict(torch.load(weights_path)['model_state_dict'])
            # model = torch.jit.load(weights_path)
            model.eval()
            # model = tta.TTAWrapper(model, flip_image2mask)
            models.append(model)
            print('loaded {}'.format(weights_path))

        file_names = sorted(valid_dataset.ids)
        for batch_i, test_batch in enumerate(tqdm(valid_loader)):
            for model_i, model in enumerate(models):
                runner_out = model(test_batch[0].cuda())
                runner_out = torch.sigmoid(runner_out)
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
                file_name = os.path.join(test_predict_result, names[i]) + '.tif'
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
