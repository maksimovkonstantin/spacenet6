import shutil
import os
import fire


def remove_non_sar(data_root_path):
    to_remove = ['PAN', 'PS-RGB', 'PS-RGBNIR', 'RGBNIR', 'SummaryData']
    for el in to_remove:
        path = os.path.join(data_root_path, el)
        shutil.rmtree(path)
    print('Non SAR data removed from train')
    

if __name__ == '__main__':
    fire.Fire(remove_non_sar)
    
    