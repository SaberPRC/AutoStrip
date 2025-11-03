import os
import sys
import gzip
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from IPython import embed

from dataset.basic import _ants_img_info, _normalize_z_score, _train_test_split, _idx_crop
from dataset.basic import _random_seed, _mask_seed, _crop_and_convert_to_tensor


class BETBase(torch.utils.data.Dataset):
    '''
    TODO: Data loader for brain extraction training
    '''
    def __init__(self, root, file_list, fold, type='train', crop_size=(128, 128, 128), BE=1):
        """
        # TODO: Parameters for data loader
        @param root: Project root
        @param file_list: file list name, default file_list.csv
        @param fold: which fold for testing
        @param type: specify training, validation, and testing
        @param crop_size: patch size for training
        @param BE: whether to using boundary preserving loss during training
        """
        super().__init__()
        self.root = root
        self.type = type
        self.crop_size = crop_size
        file_list = os.path.join(root, 'csvfile', file_list)
        self.file_list = _train_test_split(file_list, fold, type)
        self.BE = BE

    def __getitem__(self, idx):
        file_name, folder = self.file_list[idx][0], self.file_list[idx][1]

        img_path = os.path.join(self.root, 'data', folder, file_name, 'pseudo_brain.nii.gz')
        seg_path = os.path.join(self.root, 'data', folder, file_name, 'skull-strip.nii.gz')

        origin, spacing, direction, img = _ants_img_info(img_path)
        origin, spacing, direction, seg = _ants_img_info(seg_path)

        img = _normalize_z_score(img)

        # Padding for large scale data patch
        img = np.pad(img, ((16, 16), (16, 16), (16, 16)), 'constant')

        if self.type == 'train':
            seg = np.pad(seg, ((16, 16), (16, 16), (16, 16)), 'constant')
            # TODO: Functions to unbalanced patch crop according to segmentation
            if random.random() > 0.2:
                start_pos = _mask_seed(seg, self.crop_size)
            else:
                start_pos = _random_seed(seg, self.crop_size)

            img_cropped = _crop_and_convert_to_tensor(img, start_pos, self.crop_size)
            seg_cropped = _crop_and_convert_to_tensor(seg, start_pos, self.crop_size)

            if self.BE == 1:
                seg_cropped_one_hot = seg[start_pos[0]:start_pos[0] + self.crop_size[0],
                                     start_pos[1]:start_pos[1] + self.crop_size[1],
                                     start_pos[2]:start_pos[2] + self.crop_size[2]]
                seg_cropped_one_hot = torch.from_numpy(seg_cropped_one_hot).type(torch.long)
                seg_cropped_one_hot = F.one_hot(seg_cropped_one_hot, 2)
                seg_cropped_one_hot = seg_cropped_one_hot.permute(3, 0, 1, 2)
                seg_cropped_one_hot = seg_cropped_one_hot.type(torch.float32)
            else:
                seg_cropped_one_hot = 0

            return img_cropped, seg_cropped, seg_cropped_one_hot

        elif self.type == 'val' or self.type == 'test':
            img = torch.from_numpy(img).type(torch.float32)
            seg = torch.from_numpy(seg).type(torch.float32)

            return img, seg, file_name, origin, spacing, direction

    def __len__(self):
        return len(self.file_list)
