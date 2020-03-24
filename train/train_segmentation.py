import sys
import torch
import os.path as osp
from importlib import import_module
from models.model_factory import make_model
from losses import get_loss
from optimizers import get_optimizer
from catalyst.dl.runner import SupervisedRunner
from utils.helpers import get_config, parse_config_args
from catalyst.dl.callbacks import CheckpointCallback, DiceCallback, SchedulerCallback, EarlyStoppingCallback, IouCallback
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
                                                               mode='max',
                                                               threshold=1e-6,
                                                               min_lr=1e-5)
    else:
        scheduler = None
    callbacks = []
    #classes = ['Fish',
    #           'Flower',
    #           'Gravel',
    #           'Sugar']
    #callbacks.append(MultiClassDiceScoreCallback(num_classes=n_classes,
    #                                             class_names=classes,
    #                                            mode='multilabel'))
    # callbacks.append(DiceCallback())
    # callbacks.append(IouCallback())
    callbacks.append(CheckpointCallback(save_n_best=4))
    # callbacks.append(SchedulerCallback(reduce_metric='val_loss'))
    callbacks.append(EarlyStoppingCallback(patience=config['early_stopping'],
                                          metric='loss',
                                          minimize=True,
                                          min_delta=1e-6))
    
    
    
    

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
                 main_metric='loss',
                 minimize_metric=True,
                 fp16=fp16
                )
