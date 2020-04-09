import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from .unet import DPNUnet

def make_model(model_name='unet_resnet34',
               weights='imagenet',
               n_classes=2,
               input_channels=4):
    if model_name == 'selim_dpn92':
        model = DPNUnet(seg_classes=2,
                        backbone_arch='dpn92',
                        num_channels=4)
    return model


def make_model_old(model_name='unet_resnet34',
               weights='imagenet', n_classes=2, input_channels=4):
    if weights != 'imagenet' or weights is None:
        weights_to_load = None
    else:
        weights_to_load = weights

    if model_name.split('_')[0] == 'unet':
        
        model = smp.Unet('_'.join(model_name.split('_')[1:]), 
                         classes=n_classes,
                         activation=None,
                         encoder_weights=weights_to_load
                         #encoder_weights='imagenet+5k'
        )
        
    elif model_name.split('_')[0] == 'fpn':
        model = smp.FPN('_'.join(model_name.split('_')[1:]), 
                         classes=n_classes,
                         activation=None,
                         encoder_weights=weights_to_load)
    elif model_name.split('_')[0] == 'linknet':
        model = smp.Linknet('_'.join(model_name.split('_')[1:]),
                         classes=n_classes,
                         activation=None,
                         encoder_weights=weights_to_load)
    else:
        raise ValueError('Model not implemented')

    if weights != 'imagenet' and weights is not None:
        print('Pretrained weights loaded')
        model.load_state_dict(torch.load(weights)['model_state_dict'])
    #
    if input_channels != 3:
        if model_name.split('_')[1] == 'vgg11':
            # trained_kernel = model.encoder.conv1.weight
            trained_kernel = model.encoder.features[0].weight
            in_channels = input_channels
            new_conv = nn.Conv2d(
                       in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
                   )
            with torch.no_grad():
                new_conv.weight[:, :] = torch.stack(
                           [torch.mean(trained_kernel, 1)] * in_channels, dim=1
                       )
            model.encoder.features[0] = new_conv
        elif model_name.split('_')[1][:8] == 'densenet':
            trained_kernel = model.encoder.features.conv0.weight
            # trained_kernel = model.encoder.features.conv1_1.conv.weight
            in_channels = input_channels
            new_conv = nn.Conv2d(
                in_channels, 96, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                new_conv.weight[:, :] = torch.stack(
                    [torch.mean(trained_kernel, 1)] * in_channels, dim=1
                )
            model.encoder.features.conv0 = new_conv

        elif model_name.split('_')[1][:12] == 'efficientnet' or model_name.split('_')[1][:3] == 'dpn':
            model = smp.Unet('_'.join(model_name.split('_')[1:]),
                             classes=n_classes,
                             in_channels=input_channels,
                             activation=None,
                             #encoder_weights=weights_to_load
                             encoder_weights='imagenet+5k'
                             )
        else:
            print('AAAAAAAAAAAAAAAAAAA')
            trained_kernel = model.encoder.conv1.weight
            # trained_kernel = model.encoder.features.conv1_1.conv.weight
            in_channels = input_channels
            new_conv = nn.Conv2d(
                       in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                   )
            with torch.no_grad():
                new_conv.weight[:, :] = torch.stack(
                           [torch.mean(trained_kernel, 1)] * in_channels, dim=1
                       )
            model.encoder.conv1= new_conv
    
    return model
