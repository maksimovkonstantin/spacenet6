import shutil
import os
import fire

def create_masks(data_root_path,
                 result_path,
                 borders=True,
                 area_in_beetwen=True):
    
    labels_path = os.path.join(data_root_path, 'geojson_buildings')
    os.mkdir(result_path)
    
    pass

if __name__ == '__main__':
    fire.Fire(create_masks)
    