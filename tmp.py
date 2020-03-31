files = sorted(os.listdir(data_path))
f = open(submit_path, 'w')
f.write('ImageId,BuildingId,PolygonWKT_Pix,Confidence\n')
for _file in tqdm(files):
    file_path = os.path.join(data_path, _file)
    pred_data = skimage.io.imread(file_path, plugin='tifffile')
    labels = wsh(pred_data[:, :, 0] / 255., prob_trs,
                 1 - pred_data[:, :, 1] / 255.,
                 pred_data[:, :, 0] / 255., shift)
    label_numbers = list(np.unique(labels))
    all_dfs = []
    for label in label_numbers:
        if label != 0:
            submask = (labels == label).astype(np.uint8)
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
            if len(df) > 0:
                all_dfs.append(df.copy())
    # check this code again

    if len(all_dfs) > 0:
        df_poly = pd.concat(all_dfs)

        df_poly = df_poly.sort_values(by='area_size', ascending=False)
        df_poly.loc[:, 'wkt'] = df_poly.poly.apply(lambda x: shapely.wkt.dumps(
            x, rounding_precision=0))
        df_poly.loc[:, 'bid'] = list(range(1, len(df_poly) + 1))
        df_poly.loc[:, 'area_ratio'] = df_poly.area_size / df_poly.area_size.max()
        for i, row in df_poly.iterrows():
            line = "{},{},\"{}\",{:.6f}\n".format(
                fid,
                row.bid,
                row.wkt,
                row.area_ratio)
            line = _remove_interiors(line)
            f.write(line)
    else:
        f.write("{},{},{},0\n".format(
            fid,
            -1,
            "POLYGON EMPTY")
        f.close()