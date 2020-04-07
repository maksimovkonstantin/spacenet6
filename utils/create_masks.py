import shutil
import os
import fire
import gdal
import numpy as np
import cv2
from tqdm import tqdm
from osgeo import ogr


def create_masks(data_root_path='/data/SN6_buildings/train/AOI_11_Rotterdam/',
                 result_path='/wdata/train_masks/'):

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)

    labels_path = os.path.join(data_root_path, 'geojson_buildings')
    rasters_path = os.path.join(data_root_path, 'SAR-Intensity')

    files = sorted(os.listdir(labels_path))
    ids = ['_'.join(el.split('.')[0] .split('_')[6:])  for el in files]

    for _id in tqdm(ids[:]):
        label_path = os.path.join(labels_path, 'SN6_Train_AOI_11_Rotterdam_Buildings_' + _id + '.geojson')
        raster_path = os.path.join(rasters_path, 'SN6_Train_AOI_11_Rotterdam_SAR-Intensity_' + _id + '.tif')
        tileHdl = gdal.Open(raster_path, gdal.GA_ReadOnly)
        tileGeoTransformationParams = tileHdl.GetGeoTransform()
        projection = tileHdl.GetProjection()
        width = tileHdl.RasterXSize
        height = tileHdl.RasterYSize

        tileHdl = None

        rasterDriver = gdal.GetDriverByName('MEM')

        final_mask = rasterDriver.Create('',
                                         height,
                                         width,
                                         1,
                                         gdal.GDT_Byte)

        final_mask.SetGeoTransform(tileGeoTransformationParams)
        final_mask.SetProjection(projection)
        tempTile = final_mask.GetRasterBand(1)
        tempTile.Fill(0)
        tempTile.SetNoDataValue(0)

        Polys_ds = ogr.Open(label_path)
        Polys = Polys_ds.GetLayer()
        gdal.RasterizeLayer(final_mask, [1], Polys, burn_values=[255])
        mask = final_mask.ReadAsArray()
        final_mask = None

        rasterDriver = gdal.GetDriverByName('GTiff')


        out_path = os.path.join(result_path, _id + '.tif')

        final_mask = rasterDriver.Create(out_path,
                                         height,
                                         width,
                                         3,
                                         gdal.GDT_Byte)

        final_mask.SetGeoTransform(tileGeoTransformationParams)
        final_mask.SetProjection(projection)
        tempTile = final_mask.GetRasterBand(1)
        tempTile.Fill(0)
        tempTile.SetNoDataValue(0)
        tempTile.WriteArray(mask[:, :])
        h, w = mask.shape
        all_contours = np.zeros((h, w), dtype=np.uint8)
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            img = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(img, [cnt], 0, 255, 1)
            contour = img.astype(np.uint8)
            all_contours += contour
        tempTile = final_mask.GetRasterBand(2)
        tempTile.Fill(0)
        tempTile.SetNoDataValue(0)
        tempTile.WriteArray(all_contours[:, :])

        

        final_mask = None


if __name__ == '__main__':
    fire.Fire(create_masks)
