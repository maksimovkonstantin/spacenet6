import torch
import os
from utils.helpers import get_config, parse_config_args
from utils.helpers import get_config, parse_config_args
from models.model_factory import make_model


def trace(model, path_to_save, input_channels, shape):
    model.eval()
    example_forward_input = torch.rand(1, input_channels, shape[0], shape[1]).cuda()
    module = torch.jit.trace(model, example_forward_input)
    module.save(path_to_save)


def main():
    args = parse_config_args()
    config = get_config(args.config)
    weights_path = config['load_from']
    model_name = config['model_name']
    device = 'cuda'
    name_to_save = os.path.join('/wdata/traced_models/', weights_path.split('/')[-3] + '.pth')
    n_classes = config['n_classes']
    input_channels = config['input_channels']
    crop_size = config['crop_size']

    model = make_model(
        model_name=model_name,
        weights=None,
        n_classes=n_classes,
        input_channels=input_channels).to(device)

    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.eval()
    trace(model, name_to_save, input_channels, shape=crop_size)
    print('Model traced to {}'.format(name_to_save))


if __name__ == '__main__':
    main()