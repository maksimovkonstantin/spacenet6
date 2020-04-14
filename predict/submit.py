import os
import skimage.io
import numpy as np
import rasterio.features
import shapely.ops
import shapely.wkt
import shapely.geometry
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import watershed
from tqdm import tqdm

def _remove_interiors(line):
    if "), (" in line:
        line_prefix = line.split('), (')[0]
        line_terminate = line.split('))",')[-1]
        line = (
            line_prefix +
            '))",' +
            line_terminate
        )
    return line


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


prob_trs = 0.3
shift = 0.4

MIN_POLYGON_AREA = 200
submit_path = '/wdata/submits/solution.csv'
data_path = '/wdata/segmentation_test_results/'

files = sorted(os.listdir(data_path))[:]
f = open(submit_path, 'w')
f.write('ImageId,PolygonWKT_Pix,Confidence\n')
for _file in tqdm(files):
    file_path = os.path.join(data_path, _file)
    fid = '_'.join(_file.split('_')[-4:]).split('.')[0]
    # print(fid)
    pred_data = skimage.io.imread(file_path, plugin='tifffile')
    labels = wsh(pred_data[:, :, 0], prob_trs,
                 1 - pred_data[:, :, 1],
                        #(1 - pred_data[:, :, 1])*(1 - pred_data[:, :, 2]),
                         pred_data[:, :, 0],
                 shift)
    label_numbers = list(np.unique(labels))
    all_dfs = []
    for label in label_numbers:
        if label != 0:
            submask = (labels == label).astype(np.uint8)
            if np.sum(submask) < MIN_POLYGON_AREA:
                continue
            shapes = rasterio.features.shapes(submask.astype(np.int16), submask > 0)

            mp = shapely.ops.cascaded_union(
                shapely.geometry.MultiPolygon([
                    shapely.geometry.shape(shape)
                    for shape, value in shapes
                ]))

            if isinstance(mp, shapely.geometry.Polygon):
                df = pd.DataFrame({
                    'area_size': [mp.area],
                    'poly': [mp],
                })
            else:
                df = pd.DataFrame({
                    'area_size': [p.area for p in mp],
                    'poly': [p for p in mp],
                })
                # made cheanges
            df = df[df.area_size > MIN_POLYGON_AREA]
            df = df.reset_index(drop=True)
            # print(df)
            if len(df) > 0:
                all_dfs.append(df.copy())
    if len(all_dfs) > 0:
        df_poly = pd.concat(all_dfs)
        df_poly = df_poly.sort_values(by='area_size', ascending=False)
        df_poly.loc[:, 'wkt'] = df_poly.poly.apply(lambda x: shapely.wkt.dumps(x, rounding_precision=0))
        df_poly.loc[:, 'area_ratio'] = df_poly.area_size / df_poly.area_size.max()
        for i, row in df_poly.iterrows():

            line = "{},\"{}\",{:.6f}\n".format(
                fid,
                row.wkt,
                row.area_ratio)
            line = _remove_interiors(line)
            # print(line)
            f.write(line)
    else:
        print('file {} is empty'.format(fid))
        f.write("{},{},0\n".format(
            fid,
            "POLYGON EMPTY"))
f.close()