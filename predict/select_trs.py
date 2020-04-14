import os
import numpy as np
import skimage.io
from scipy import ndimage as ndi
from skimage.morphology import watershed
from tqdm import tqdm
import matplotlib.pyplot as plt


def my_watershed(what, mask1, mask2):
    markers = ndi.label(mask2, output=np.uint32)[0]
    labels = watershed(what, markers, mask=mask1, watershed_line=True)
    return labels


def wsh(mask_img, threshold, border_img, seeds, shift):
    img_copy = np.copy(mask_img)
    m = seeds * border_img

    img_copy[m <= threshold + shift] = 0
    img_copy[m > threshold + shift] = 1
    img_copy = img_copy.astype(np.bool)

    mask_img[mask_img <= threshold] = 0
    mask_img[mask_img > threshold] = 1
    mask_img = mask_img.astype(np.bool)
    labeled_array = my_watershed(mask_img, mask_img, img_copy)
    return labeled_array


def calc_score(labels, y_pred):
    if y_pred.sum() == 0 and labels.sum() == 0:
        return 1
        print('AAAAA')
    if labels.sum() == 0 and y_pred.sum() > 0 or y_pred.sum() == 0 and labels.sum() > 0:
        return 0

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    return precision_at(0.5, iou)


# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

gt_path = '/wdata/train_masks/'
pred_path = '/wdata/segmentation_validation_results/'

ids = sorted(os.listdir(pred_path))

prob_trs = 0.3
shift = 0.3
size = 400
summary = [0, 0, 0]
# probs = [0.1, 0.2, 0.3, 0.4, 0.45]
# shifts = [0.1, 0.2, 0.3, 0.4, 0.45]
probs = [0.3, 0.4, 0.5, 0.6]
# probs = [0.7, 0.8, 0.9]
shifts = [0.2, 0.3, 0.4]
#probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#shifts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_params = ''
best_res = 0
for prob_trs in probs:
    for shift in shifts:
        if prob_trs + shift >= 1.0:
            continue
        summary = [0, 0, 0]
        for _file in tqdm(ids[:100]):
            pred_data = skimage.io.imread(os.path.join(pred_path, _file), plugin='tifffile')
            gt = (skimage.io.imread(os.path.join(gt_path, _file), plugin='tifffile')[:, :, 0] / 255).astype(np.uint8)
            gt_labels = ndi.label(gt, output=np.uint32)[0]
            # pred_data[:, :, 1] = 0.0
            tmp_labels = wsh(pred_data[:, :, 0], prob_trs,
                             # (1 - pred_data[:, :, 2]) * (1 - pred_data[:, :, 1]),
                             # np.ones(pred_data.shape[:2]),
                             # (1 - pred_data[:, :, 1]),
                             (1 - pred_data[:, :, 1]),
                             pred_data[:, :, 0], shift)
            # plt.imshow(tmp_labels)
            pred_labels = tmp_labels
            # pred_labels = np.zeros(tmp_labels.shape, dtype=np.uint32)
            # new_label_count = 0
            # for pred_label in pred_labels:
            #    if np.sum(tmp_labels == pred_label) > size:
            #        pred_labels[tmp_labels == pred_label] = new_label_count
            #        new_label_count += 1
            # print(calc_score)
            # print(calc_score(gt_labels, pred_labels))
            # tp, fp, fn = calc_score(gt_labels, pred_labels)
            res = calc_score(gt_labels, pred_labels)
            if isinstance(res, tuple):
                tp, fp, fn = res
                summary[0] += tp
                summary[1] += fp
                summary[2] += fn
        tp, fp, fn = summary
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        if (precision + recall) > 0:

            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        print('prob {} shift {} result '.format(prob_trs, shift), (tp, fp, fn), precision, recall, f1)
        if f1 > best_res:
            best_res = f1
            best_params = 'prob {} shift {} result '.format(prob_trs, shift)

        # plt.imshow(predict[:, :, 0])
print(best_res, best_params)
