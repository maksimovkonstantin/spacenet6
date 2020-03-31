import torch
from tqdm import tqdm
import os
import numpy as np
from utils.helpers import get_config, parse_config_args
from models.model_factory import make_model
import skimage.io
import albumentations as albu
import rasterio

if __name__ == '__main__':
    with torch.no_grad():
        args = parse_config_args()
        config = get_config(args.config)
        config_loaders = config['loaders']
        model_name = config['model_name']
        weights_path = config['load_from']
        device = config['device']
        loss_name = config['loss']
        fold = config['fold_number']
        model_save_name = '_'.join(config['model_name'].split('_')[1:])
        val_batch_size = config['val_batch_size']
        input_channels = config['input_channels']
        test_loader = config['valid_loader']
        n_classes = config['n_classes']
        test_predict_result = config['validation_predict_result']
        original_size = config['original_size']
        cropper = albu.Compose([albu.CenterCrop(original_size[0], original_size[1], p=1.0)])
        model = make_model(
            model_name=model_name,
            weights=None,
            n_classes=n_classes,
            input_channels=input_channels).to(device)

        model.load_state_dict(torch.load(weights_path)['model_state_dict'])
        model.eval()

        file_names = sorted(config['valid_dataset'].ids)
        # runner = SupervisedRunner(model=model)

        for batch_i, test_batch in enumerate(tqdm(test_loader)):
            # print(test_batch.shape)
            # runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
            runner_out = model(test_batch[0].cuda())
            image_pred = torch.sigmoid(runner_out)
            image_pred = image_pred.cpu().detach().numpy()
            names = file_names[batch_i * val_batch_size:(batch_i + 1) * val_batch_size]
            for i in range(len(names)):
                file_name = os.path.join(test_predict_result, names[i] + '.tif')
                # data = image_pred[i, ...][0, ...]
                data = image_pred[i, ...]
                data = np.moveaxis(data, 0, -1)
                sample = cropper(image=data)

                data = sample['image']
                data = np.moveaxis(data, -1, 0)
                # print(data.shape)
                c, h, w = data.shape
                with rasterio.open(file_name, "w", dtype=rasterio.float32, driver='GTiff',
                                   width=h, height=h, count=c) as dest:
                    dest.write(data)
                # skimage.io.imsave(file_name, data, plugin='tifffile')
                # print(data.shape)
