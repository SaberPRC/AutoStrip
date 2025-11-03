import os
import ants
import numpy as np
import pandas as pd
from skimage import measure
from scipy import ndimage

from tqdm import tqdm

# moving_img_path = '/public_bme/home/liujm/ReleasePackage/AutoBET/Atlas/Atlas_Adult_T1.nii.gz'
# moving_seg_path = '/public_bme/home/liujm/ReleasePackage/AutoBET/Atlas/Atlas_Adult_Skull_Strip.nii.gz'

moving_img_path = '/public_bme/home/liujm/Public/Atlas/infantile/img-T1.nii.gz'
moving_seg_path = '/public_bme/home/liujm/Public/Atlas/infantile/seg-T1.nii.gz'


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _ants_registration(moving_img_path, moving_seg_path=None, fixed_img_path=None):
    moving_img = ants.image_read(moving_img_path)
    if moving_seg_path == None:
        moving_img = ants.image_read(moving_img_path)
        fixed_img = ants.image_read(fixed_img_path)
        res = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform='SyN')
        return res['warpedmovout']
    else:
        moving_img = ants.image_read(moving_img_path)
        moving_seg = ants.image_read(moving_seg_path)
        fixed_img = ants.image_read(fixed_img_path)

        res = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform='SyN')

        warped_img = res['warpedmovout']
        warped_seg = ants.apply_transforms(fixed=fixed_img, moving=moving_seg, transformlist=res['fwdtransforms'],
                                           interpolator='nearestNeighbor')

        return warped_img, warped_seg


def _persudo_brain_extraction(source, target, item):
    source_img_path = os.path.join(source, item, 'T2.nii.gz')
    origin, spacing, direction, img = _ants_img_info(source_img_path)
    source_seg_path = os.path.join(source, item, 'skull-strip.nii.gz')
    origin, spacing, direction, persudo_mask = _ants_img_info(source_seg_path)
    persudo_mask = ndimage.binary_dilation(persudo_mask, iterations=5)
    persudo_mask = persudo_mask.astype(np.float32)
    target_img_path = os.path.join(target, item, 'T2_persudo_brain.nii.gz')

    persudo_brain = img * persudo_mask
    persudo_brain = ants.from_numpy(persudo_brain, origin, spacing, direction)

    ants.image_write(persudo_brain, target_img_path)


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


if __name__ == '__main__':
    from multiprocessing import Pool
    from tqdm import tqdm

    source = '/public_bme/data/jiameng/CBCP/CBCP_20250403_Process_All/acpc_space'
    target = '/public_bme/data/jiameng/CBCP/CBCP_20250403_Process_All/acpc_space'

    file_list = os.listdir(source)
    file_list.sort()
    # file_list = [f for f in file_list if not os.path.exists(os.path.join(source, f, 'persudo_brain.nii.gz'))]
    pool_num = 24
    pool = Pool(pool_num)
    pbar = tqdm(total=len(file_list))
    pbar.set_description('Persudo Brain Extraction')
    call_fun = lambda *args: update(pbar, *args)

    for item in file_list:
        kwargs = {
            'source': source,
            'target': target,
            'item':item
        }
        pool.apply_async(_persudo_brain_extraction, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()


