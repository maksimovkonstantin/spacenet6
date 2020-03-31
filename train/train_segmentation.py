import sys
import torch
import os.path as osp
from importlib import import_module
from models.model_factory import make_model
from losses import get_loss
from optimizers import get_optimizer
from catalyst.dl.runner import SupervisedRunner
from utils.helpers import get_config, parse_config_args
from catalyst.dl.callbacks import CheckpointCallback, SchedulerCallback, EarlyStoppingCallback, IouCallback
from callbacks import DiceCallback
# from pytorch_toolbelt.utils.catalyst.metrics import IoUMetricsCallback
# from segmentation_metrics import MultiClassDiceScoreCallback

if __name__ == '__main__':
    args = parse_config_args()
    config = get_config(args.config)
    model_name = config['model_name']
    # n_classes = config['n_classes']
    fold_number = config['fold_number']
    alias = config['alias']
    log_path = osp.join(config['logs_path'],
                        alias + str(fold_number) + '_' + model_name)
        
    device = torch.device(config['device'])
    
    loaders = config['loaders']
    
    weights = config['weights']
    # activation = config['activation']
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
    mode = config['scheduler_mode']
    min_lr = config['min_lr']
    threshold = config['thershold']
    best_models_count = config['best_models_count']
    minimize_metric = config['minimize_metric']
    min_delta = config['min_delta']

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
        alpha = config['alpha']
        patience = config['patience']
        alpha = config['alpha']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               factor=alpha,
                                                               verbose=True,
                                                               patience=patience,
                                                               mode=mode,
                                                               threshold=threshold,
                                                               min_lr=min_lr)
    else:
        scheduler = None
    callbacks = []
    # dice_callback = IoUMetricsCallback(mode='binary', lasses_of_interest=[0])
    dice_callback = DiceCallback()
    callbacks.append(dice_callback)
    callbacks.append(CheckpointCallback(save_n_best=best_models_count))
    callbacks.append(EarlyStoppingCallback(patience=config['early_stopping'],
                                          metric=main_metric,
                                          minimize=minimize_metric,
                                          min_delta=min_delta))
    
    
    
    

    runner = SupervisedRunner(device=device)

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
