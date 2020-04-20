import torch
import os.path as osp
from models.model_factory import make_model
from losses import get_loss
from optimizers import get_optimizer
from catalyst.dl.runner import SupervisedRunner
from utils.helpers import get_config, parse_config_args
from catalyst.dl.callbacks import CheckpointCallback, EarlyStoppingCallback
from callbacks import DiceCallback
from dataset.semseg_dataset import SemSegDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    args = parse_config_args()
    config = get_config(args.config)
    model_name = config['model_name']
    fold_number = config['fold_number']
    alias = config['alias']
    log_path = osp.join(config['logs_path'],
                        alias + str(fold_number) + '_' + model_name)
        
    device = torch.device(config['device'])
    weights = config['weights']
    loss_name = config['loss']
    optimizer_name = config['optimizer']
    lr = config['lr']
    decay = config['decay']
    momentum = config['momentum']
    epochs = config['epochs']
    fp16 = config['fp16']
    n_classes = config['n_classes']
    input_channels = config['input_channels']
    main_metric = config['main_metric']


    best_models_count = config['best_models_count']
    minimize_metric = config['minimize_metric']
    min_delta = config['min_delta']

    train_images = config['train_images']
    data_type = config['data_type']
    masks_data_path = config['masks_data_path']
    folds_file = config['folds_file']
    train_augs = config['train_augs']
    preprocessing_fn = config['preprocessing_fn']
    limit_files = config['limit_files']
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    num_workers = config['num_workers']
    valid_augs = config['valid_augs']
    val_batch_size = config['val_batch_size']
    multiplier = config['multiplier']

    train_dataset = SemSegDataset(images_dir=train_images,
                                  data_type=data_type,
                                  masks_dir=masks_data_path,
                                  mode='train',
                                  folds_file=folds_file,
                                  fold_number=fold_number,
                                  augmentation=train_augs,
                                  preprocessing=preprocessing_fn,
                                  limit_files=limit_files,
                                  multiplier=multiplier)

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

    model = make_model(
               model_name=model_name,
               weights=weights,
               n_classes=n_classes,
               input_channels=input_channels).to(device)

    loss = get_loss(loss_name=loss_name)
    optimizer = get_optimizer(optimizer_name=optimizer_name,
                              model=model,
                              lr=lr,
                              momentum=momentum,
                              decay=decay)

    if config['scheduler'] == 'reduce_on_plateau':
        print('reduce lr')
        alpha = config['alpha']
        patience = config['patience']
        threshold = config['thershold']
        min_lr = config['min_lr']
        mode = config['scheduler_mode']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               factor=alpha,
                                                               verbose=True,
                                                               patience=patience,
                                                               mode=mode,
                                                               threshold=threshold,
                                                               min_lr=min_lr)
    elif config['scheduler'] == 'steps':
        print('steps lr')
        steps = config['steps']
        step_gamma = config['step_gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=steps, gamma=step_gamma)
    callbacks = []

    dice_callback = DiceCallback()
    callbacks.append(dice_callback)
    callbacks.append(CheckpointCallback(save_n_best=best_models_count))
    callbacks.append(EarlyStoppingCallback(patience=config['early_stopping'],
                                           metric=main_metric,
                                           minimize=minimize_metric,
                                           min_delta=min_delta))

    runner = SupervisedRunner(device=device)
    loaders = {'train': train_loader, 'valid': valid_loader}

    runner.train(model=model,
                 criterion=loss,
                 optimizer=optimizer,
                 loaders=loaders,
                 scheduler=scheduler,
                 callbacks=callbacks,
                 logdir=log_path,
                 num_epochs=epochs,
                 verbose=True,
                 main_metric=main_metric,
                 minimize_metric=minimize_metric,
                 fp16=fp16
                )

