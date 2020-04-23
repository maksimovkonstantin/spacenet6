import os
import wget
from helpers import get_config
import gdown

url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vU3VUVlFnbTgtS2c'
output = 'spam.txt'


def main():
    models_path = '/wdata/segmentation_logs/'
    if len(os.listdir(models_path)) == 0:
        print('Loading pretrained models')
        configs = ['/project/configs/densenet161_gcc_fold1.py',
                   '/project/configs/densenet161_gcc_fold2.py',
                   '/project/configs/densenet161_gcc_fold3.py',
                   '/project/configs/densenet161_gcc_fold4.py',

                   '/project/configs/dpn92_gcc_fold1.py',
                   '/project/configs/dpn92_gcc_fold2.py',
                   '/project/configs/dpn92_gcc_fold3.py',
                   '/project/configs/dpn92_gcc_fold4.py',

                   '/project/configs/effnetb7_gcc_fold1.py',
                   '/project/configs/effnetb7_gcc_fold2.py',
                   '/project/configs/effnetb7_gcc_fold3.py',
                   '/project/configs/effnetb7_gcc_fold4.py',

                   '/project/configs/senet154_gcc_fold1.py',
                   '/project/configs/senet154_gcc_fold2.py',
                   '/project/configs/senet154_gcc_fold3.py',
                   '/project/configs/senet154_gcc_fold4.py'
                   ]

        urls = ['https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',

                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',

                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',

                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id=',
                'https://drive.google.com/uc?id='
                ]

        for config_number, config in enumerate(configs):
            create_folder = get_config(config)['load_from'].split('/')[:-1]
            print('Loading model to {}'.format(create_folder))
            os.makedirs(create_folder)
            url = urls[config_number]
            # wget.download(url, out=os.path.join(create_folder, 'best.pth'))
            gdown.download(url, os.path.join(create_folder, 'best.pth'), quiet=False)
    else:
        print('models available')


if __name__ == '__main__':
    main()