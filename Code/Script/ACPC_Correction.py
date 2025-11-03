import os
import ants

from tqdm import tqdm
from IPython import embed


def _data_reorient(source, target, item):
    target_folder = os.path.join(target, item)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    source_img_path = os.path.join(source, item, 'T2.nii.gz')
    target_img_path = os.path.join(target_folder, 'T2.nii.gz')
    img = ants.image_read(source_img_path)
    nimg = ants.reorient_image2(img, 'RPI')
    ants.image_write(nimg, target_img_path)

    source_img_path = os.path.join(source, item, 'skull-strip.nii.gz')
    target_img_path = os.path.join(target_folder, 'skull-strip.nii.gz')
    img = ants.image_read(source_img_path)
    nimg = ants.reorient_image2(img, 'RPI')
    ants.image_write(nimg, target_img_path)

    source_img_path = os.path.join(source, item, 'T2.nii.gz')
    target_img_path = os.path.join(target_folder, 'persudo_brain.nii.gz')
    os.system('cp {} {}'.format(source_img_path, target_img_path))


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


if __name__ == '__main__':
    from multiprocessing import Pool
    from tqdm import tqdm
    from IPython import embed

    source = '/public_bme2/bme-dgshen/JiamengLiu/BET/data/T2/CBCP'
    target = '/public_bme2/bme-dgshen/JiamengLiu/BET/data/T2/CBCP'

    file_list = os.listdir(source)
    file_list.sort()

    pool_num = 16
    pool = Pool(pool_num)
    pbar = tqdm(total=len(file_list))
    pbar.set_description('Data Reorient')
    call_fun = lambda *args: update(pbar, *args)

    for item in file_list:
        kwargs = {
            'source':source,
            'target':target,
            'item':item
            }
        pool.apply_async(_data_reorient, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()

