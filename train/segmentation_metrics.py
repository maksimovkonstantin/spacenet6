import numpy as np
import torch
from catalyst.dl import Callback, RunnerState, MetricCallback
from functools import partial

def binary_dice_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0.5, eps=1e-3):
    # Binarize predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)
    #if y_true.sum() == 0:
    #    return 1
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = 2 * intersection / (union + eps)*100
    return float(dice)


def multiclass_dice_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0.5, eps=1e-3, classes_of_interest=None):
    dices = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.sigmoid()

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        dice = binary_dice_score((y_true == class_index).float(),
                               (y_pred == class_index).float(), threshold, eps)
        dices.append(dice)

    return dices


def multilabel_dice_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0.5, eps=1e-3, classes_of_interest=None):
    dices = []
    num_classes = y_pred.size(0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        dice = binary_dice_score(y_true[class_index], y_pred[class_index], threshold, eps)
        dices.append(dice)

    return dices


class MultiClassDiceScoreCallback(Callback):
    """
    Jaccard metric callback which is computed across whole epoch, not per-batch.
    """

    def __init__(self,
                 mode: str,
                 num_classes: int = None,
                 class_names=None,
                 from_logits=True,
                 classes_of_interest=None,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "dice"):
        """
        :param mode: One of: 'binary', 'multiclass', 'multilabel'.
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        assert mode in {'binary', 'multiclass', 'multilabel'}

        if classes_of_interest is not None:
            if classes_of_interest.dtype == np.bool:
                num_classes = len(classes_of_interest)
                classes_of_interest = np.arange(num_classes)[classes_of_interest]

        self.mode = mode
        self.prefix = prefix
        self.from_logits = from_logits
        self.output_key = output_key
        self.input_key = input_key
        self.class_names = class_names
        self.classes_of_interest = classes_of_interest
        self.scores = []

        if self.mode == 'binary':
            self.score_fn = binary_dice_score

        if self.mode == 'multiclass':
            self.score_fn = partial(multiclass_dice_score, classes_of_interest=self.classes_of_interest)

        if self.mode == 'multilabel':
            self.score_fn = partial(multilabel_dice_score, classes_of_interest=self.classes_of_interest)

    def on_loader_start(self, state):
        self.scores = []
    
    @property
    def order(self):
        return 1
    
    
    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key].detach()
        if self.from_logits:
            outputs = outputs.sigmoid()
        targets = state.input[self.input_key].detach()
        batch_size = targets.size(0)
        dices = []
        for image_index in range(batch_size):
            dice_per_class = self.score_fn(targets[image_index], outputs[image_index])
            dices.append(dice_per_class)

        dice_per_batch = np.nanmean(dices)
        state.metrics.add_batch_value(self.prefix, float(dice_per_batch))
        self.scores.extend(dices)

    def on_loader_end(self, state):
        scores = np.array(self.scores)
        dice = np.nanmean(scores)

        state.metrics.epoch_values[state.loader_name][self.prefix] = float(dice)

        # Log additional Dice scores per class
        if self.mode in {'multiclass', 'multilabel'}:
            num_classes = scores.shape[1]
            class_names = self.class_names
            if class_names is None:
                class_names = [f'class_{i}' for i in range(num_classes)]

            scores_per_class = np.nanmean(scores, axis=0)
            for class_name, score_per_class in zip(class_names, scores_per_class):
                state.metrics.epoch_values[state.loader_name][self.prefix + '_' + class_name] = float(score_per_class)

def binary_iou_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0., eps=1e-3):
    if y_true.sum() == 0:
        return np.nan

    # Binarize predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    iou = intersection / (union - intersection + eps)
    return float(iou)


def multiclass_iou_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0., eps=1e-3, classes_of_interest=None):
    ious = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.argmax(dim=0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_iou_score((y_true == class_index).float(),
                               (y_pred == class_index).float(), threshold, eps)
        ious.append(iou)

    return ious


def multilabel_iou_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0., eps=1e-3, classes_of_interest=None):
    ious = []
    num_classes = y_pred.size(0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_iou_score(y_true[class_index], y_pred[class_index], threshold, eps)
        ious.append(iou)

    return ious


class JaccardScoreCallback(Callback):
    """
    Jaccard metric callback which is computed across whole epoch, not per-batch.
    """

    def __init__(self,
                 mode: str,
                 num_classes: int = None,
                 from_logits=True,
                 class_names=None,
                 classes_of_interest=None,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "jaccard"):
        """
        :param mode: One of: 'binary', 'multiclass', 'multilabel'.
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        assert mode in {'binary', 'multiclass', 'multilabel'}

        if classes_of_interest is not None:
            if classes_of_interest.dtype == np.bool:
                num_classes = len(classes_of_interest)
                classes_of_interest = np.arange(num_classes)[classes_of_interest]

            if class_names is not None:
                if len(class_names) != len(classes_of_interest):
                    raise ValueError('Length of \'classes_of_interest\' must be equal to length of \'classes_of_interest\'')

        self.mode = mode
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.class_names = class_names
        self.from_logits = from_logits
        self.classes_of_interest = classes_of_interest
        self.scores = []

        if self.mode == 'binary':
            self.score_fn = binary_iou_score

        if self.mode == 'multiclass':
            self.score_fn = partial(multiclass_iou_score, classes_of_interest=self.classes_of_interest)

        if self.mode == 'multilabel':
            self.score_fn = partial(multilabel_iou_score, classes_of_interest=self.classes_of_interest)

    def on_loader_start(self, state):
        self.scores = []

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key].detach()
        if self.from_logits:
            outputs = outputs.softmax(dim=1)
        targets = state.input[self.input_key].detach()

        batch_size = targets.size(0)
        ious = []
        for image_index in range(batch_size):
            iou_per_class = self.score_fn(targets[image_index], outputs[image_index])
            ious.append(iou_per_class)

        iou_per_batch = np.nanmean(ious)
        state.metrics.add_batch_value(self.prefix, float(iou_per_batch))
        self.scores.extend(ious)

    def on_loader_end(self, state):
        scores = np.array(self.scores)
        iou = np.nanmean(scores)

        state.metrics.epoch_values[state.loader_name][self.prefix] = float(iou)

        # Log additional IoU scores per class
        if self.mode in {'multiclass', 'multilabel'}:
            num_classes = scores.shape[1]
            class_names = self.class_names
            if class_names is None:
                class_names = [f'class_{i}' for i in range(num_classes)]

            scores_per_class = np.nanmean(scores, axis=0)
            for class_name, score_per_class in zip(class_names, scores_per_class):
                state.metrics.epoch_values[state.loader_name][self.prefix + '_' + class_name] = float(score_per_class)

                
