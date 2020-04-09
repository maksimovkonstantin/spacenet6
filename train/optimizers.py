import torch
from radam import RAdam


def get_optimizer(optimizer_name, model, lr, momentum, decay):
    if optimizer_name == 'radam':
        
        optimizer = RAdam(model.parameters(),
                                    lr,
                                    weight_decay=decay)
    elif optimizer_name == 'adam':
        
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr,
                                    weight_decay=decay)
        
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr,
                                    momentum=momentum,
                                    weight_decay=decay)
    
    else:
        optimizer = None
    return optimizer
