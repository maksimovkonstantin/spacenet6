import os
import shutil
from utils.helpers import get_config

def main():
    configs = ['../configs/densenet161_gcc_fold1.py',
               '../configs/densenet161_gcc_fold2.py',
               #'/project/configs/densenet161_gcc_fold3.py',
               #'/project/configs/densenet161_gcc_fold4.py',

               #'/project/configs/dpn92_gcc_fold1.py',
               #'/project/configs/dpn92_gcc_fold2.py',
               #'/project/configs/dpn92_gcc_fold3.py',
               #'/project/configs/dpn92_gcc_fold4.py',

               #'/project/configs/effnetb7_gcc_fold1.py',
               #'/project/configs/effnetb7_gcc_fold2.py',
               #'/project/configs/effnetb7_gcc_fold3.py',
               #'/project/configs/effnetb7_gcc_fold4.py',

               #'/project/configs/senet154_gcc_fold1.py',
               #'/project/configs/senet154_gcc_fold2.py',
               #'/project/configs/senet154_gcc_fold3.py',
               #'/project/configs/senet154_gcc_fold4.py'
               ]

    for config_number, config in enumerate(configs):

        data = get_config(config)
        src = data['load_from']
        print('Copying of {}'.format(src))
        dst = os.path.join('/wdata/final_models/', src.split('/')[-3] + '.pth')
        shutil.copy(src, dst)


if __name__ == '__main__':
    main()
