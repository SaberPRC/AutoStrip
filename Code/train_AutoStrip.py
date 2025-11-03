'''
Repo. for train the synthesis based isointense phase infant brain tissue segmentation network
Using the Modified V-Net and supervised by focal & dice loss
Copy right: Jiameng Liu, ShanghaiTech University
Contact: JiamengLiu.PRC@gmail.com
'''

import os
import sys

import ants
import time
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from IPython import embed
from config.config import cfg
from network.model import ASNet
from torch.utils.data import DataLoader
from dataset.dataset import BETBase
from utils.loss import DiceLoss, FocalLoss, soft_cldice
from torch.utils.tensorboard import SummaryWriter
from utils.utils import set_initial, calculate_patch_index, weights_init, logging_init


def get_pred(img, model, batch_size):
    if len(img.shape) == 4:
        img = torch.unsqueeze(img, dim=0)

    B, C, W, H, D = img.shape

    m = nn.ConstantPad3d(16, 0)

    pos = calculate_patch_index((W, H, D), batch_size, overlap_ratio=0.4)
    pred_rec_s = torch.zeros((cfg.dataset.num_classes+1, W, H, D))

    freq_rec = torch.zeros((cfg.dataset.num_classes+1, W, H, D))

    for start_pos in pos:
        patch = img[:,:,start_pos[0]:start_pos[0]+batch_size[0], start_pos[1]:start_pos[1]+batch_size[1], start_pos[2]:start_pos[2]+batch_size[2]]
        model_out_s,_ = model(patch)

        model_out_s = m(model_out_s)

        model_out_s = model_out_s.cpu().detach()

        pred_rec_s[:, start_pos[0]:start_pos[0]+batch_size[0], start_pos[1]:start_pos[1]+batch_size[1], start_pos[2]:start_pos[2]+batch_size[2]] += model_out_s[0,:,:,:,:]
        freq_rec[:, start_pos[0]:start_pos[0]+batch_size[0], start_pos[1]:start_pos[1]+batch_size[1], start_pos[2]:start_pos[2]+batch_size[2]] += 1

    pred_rec_s = pred_rec_s / freq_rec

    pred_rec_s = pred_rec_s[:, 16:W-16, 16:H-16, 16:D-16]

    return pred_rec_s


def _multi_layer_dice_coefficient(source, target, ep=1e-8):
    '''
    TODO: functions to calculate dice coefficient of multi class
    :param source: numpy array (Prediction)
    :param target: numpy array (Ground-Truth)
    :return: vector of dice coefficient
    '''
    class_num = target.max()+1

    source = source.astype(int)
    source = np.eye(class_num)[source]
    source = source[:,:,:,1:]
    source = source.reshape((-1, class_num-1))

    target = target.astype(int)
    target = np.eye(class_num)[target]
    target = target[:,:,:,1:]
    target = target.reshape(-1, class_num-1)

    intersection = 2 * np.sum(source * target, axis=0) + ep
    union = np.sum(source, axis=0) + np.sum(target, axis=0) + ep

    return intersection / union


def test(args, model, infer_data, infer_num, epoch, device = torch.device('cuda')):
    # initial model and set parameters
    model.eval()

    # setting save_path
    save_path = os.path.join(cfg.general.save_dir, 'pred', 'chk_'+str(epoch))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rec = list()

    for idx in tqdm(range(infer_num)):
        # Get testing data info
        img, seg, img_name, origin, spacing, direction = infer_data.__getitem__(idx)

        img = img.to(device)
        img = img.unsqueeze(0)

        pred_s = get_pred(img, model, cfg.dataset.crop_size)
        pred_s = pred_s.argmax(0)
        pred_s = pred_s.numpy().astype(np.float32)

        seg = seg.numpy().astype(int)

        dice_all = _multi_layer_dice_coefficient(pred_s, seg)

        ants_img_pred_seg = ants.from_numpy(pred_s, origin, spacing, direction)

        temp = [img_name]
        temp.extend(dice_all)
        temp.append(np.mean(np.array(dice_all)))

        rec.append(temp)

        ants.image_write(ants_img_pred_seg, os.path.join(save_path, img_name + '_seg.nii.gz'))
    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(save_path, str(epoch) + '.csv'), index=False)


def train(cfg, args):
    # set initial checkpoint and testing results save path
    if cfg.general.resume_epoch == -1:
        set_initial(cfg)

    # set initial tensorboard save path
    if not os.path.exists(os.path.join(cfg.general_server.save_dir, 'log')):
        os.mkdir(os.path.join(cfg.general_server.save_dir, 'log'))
    writer = SummaryWriter(os.path.join(cfg.general_server.save_dir, 'log'))

    logging_init(args.log_name, PARENT_DIR=os.path.join(cfg.general_server.save_dir, 'log'))
    logging.info('file list for training and validation: {}'.format(cfg.general.file_list))
    logging.info('file path to save all images: {}'.format(cfg.general.save_dir))
    logging.info('project level path: {}'.format(cfg.general.root))

    # Default tensor type
    torch.set_default_dtype(torch.float32)

    # Set computing device cpu or gpu
    device = torch.device('cuda')

    # Set numpy and torch seeds
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(cfg.general.seed)

    training_set = BETBase(cfg.general.root, cfg.general.file_list, crop_size=cfg.dataset.crop_size, fold=args.fold, type='train', BE=args.BE)
    training_loader = DataLoader(training_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    infer_data = BETBase(cfg.general.root, cfg.general.file_list, crop_size=cfg.dataset.crop_size, fold=args.fold, type='val', BE=args.BE)
    infer_num = infer_data.__len__()

    # TODO: Init resume_spoch == -1, train from scratch
    if cfg.general.resume_epoch == -1:
        model = ASNet(cfg.dataset.num_modalities, cfg.dataset.num_classes + 1)
        weights_init(model)
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model = ASNet(cfg.dataset.num_modalities, cfg.dataset.num_classes + 1)
        model_path = os.path.join(cfg.general_server.save_dir, 'checkpoints', 'chk_' + str(cfg.general.resume_epoch) + '.pth.gz')
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path))

    # TODO: Optimization setting
    lr = 1e-3

    # TODO: Loss function
    loss_focal = FocalLoss(cfg.dataset.num_classes + 1)
    loss_dice = DiceLoss()
    loss_soft_dice = soft_cldice()

    for epoch in range(cfg.general.resume_epoch + 1, cfg.train.num_epochs):
        # TODO: training process for single epoch
        if epoch <= 20:
            lr = lr
        elif epoch > 20 and epoch <= 200:
            lr = 5e-4
        elif epoch > 200:
            lr = 1e-4

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

        model.train()
        print('Training epoch {}/{}'.format(epoch, cfg.train.num_epochs), end=' ~ ')

        for idx, (img, seg, seg_one_hot) in enumerate(training_loader):
            try:
                x, y = img.to(device), seg.to(device)

                B, C, W, H, D = y.shape

                y_s = y[:, :, 16:W - 16, 16:H - 16, 16:D - 16] # ground-truth for centering patch
                y_b = F.interpolate(y, size=[W - 32, H - 32, D - 32]) # ground-truth for neighborhood patch

                y_s = y_s.squeeze(1)
                y_b = y_b.squeeze(1)

                optimizer.zero_grad()

                pred_s, pred_b = model(x)

                loss_seg_dice_s = loss_dice(pred_s, y_s)
                loss_seg_focal_s = loss_focal(pred_s, y_s)

                loss_seg_dice_b = loss_dice(pred_b, y_b)
                loss_seg_focal_b = loss_focal(pred_b, y_b)

                if args.BE == 1:
                    # TODO: whether using boundary preserving loss
                    seg_one_hot_s = seg_one_hot[:, :, 16:W - 16, 16:H - 16, 16:D - 16]
                    seg_one_hot_s = seg_one_hot_s.to(device)
                    loss_seg_cldice_s = loss_soft_dice(pred_s, seg_one_hot_s)

                    seg_one_hot_b = F.interpolate(seg_one_hot, size=[W - 32, H - 32, D - 32])
                    seg_one_hot_b = seg_one_hot_b.to(device)
                    loss_seg_cldice_b = loss_soft_dice(pred_b, seg_one_hot_b)

                    loss = 0.7 * (loss_seg_dice_s + loss_seg_focal_s * 10 + loss_seg_cldice_s * 0.01) + 0.3 * (
                            loss_seg_dice_b + loss_seg_focal_b * 10 + loss_seg_cldice_b * 0.01)
                    msg = 'epoch: {}, batch: {}, learning_rate: {}, loss: {:.4f}, loss_seg_dice_s: {:.4f}, loss_seg_focal_s: {:.4f}, loss_seg_cldice_s: {:.4f}' \
                        .format(epoch, idx, optimizer.param_groups[0]['lr'], loss.item(), loss_seg_dice_s.item(),
                                loss_seg_focal_s.item(), loss_seg_cldice_s.item())
                else:
                    loss = 0.7 * (loss_seg_dice_s + loss_seg_focal_s * 10) + 0.3 * (loss_seg_dice_b + loss_seg_focal_b * 10)
                    msg = 'epoch: {}, batch: {}, learning_rate: {}, loss: {:.4f}, loss_seg_dice_s: {:.4f}, loss_seg_focal_s: {:.4f}' \
                        .format(epoch, idx, optimizer.param_groups[0]['lr'], loss.item(), loss_seg_dice_s.item(),
                                loss_seg_focal_s.item())

                loss.backward()
                optimizer.step()
                logging.info(msg)

                if epoch != 0 and epoch % cfg.train.save_epoch == 0:
                    if idx == 0 and epoch >= 1:
                        save_path = os.path.join(cfg.general_server.save_dir, 'checkpoints', 'chk_' + str(epoch) + '.pth.gz')
                        torch.save(model.state_dict(), save_path)
                        test(args, model, infer_data, infer_num, epoch)
            except:
                print('errors')

        print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting for Brain Extraction')
    parser.add_argument('--fold', type=bool, default=1, help='specify testing fold')
    parser.add_argument('--save_path', type=str, default='AutoStrip', help='specify results save folder')
    parser.add_argument('--file_list', type=str, default='file_list.csv', help='specify training file list')
    parser.add_argument('--resume', type=int, default=-1, help='training from scratch or resume from pretrained checkpoints. default -1: training from scratch')
    parser.add_argument('--batch_size', type=int, default=1, help='# of batch size')
    parser.add_argument('--log_name', type=str, default='bet_list', help='log name for saving log info')
    parser.add_argument('--BE', type=int, default=1, help='boundary loss')

    args = parser.parse_args()

    cfg.general_server.file_list = args.file_list
    cfg.general_server.save_dir = os.path.join(cfg.general_server.save_root, args.save_path)

    cfg.dataset.crop_size = [160, 160, 160]
    cfg.general.resume_epoch = args.resume
    cfg.train.batch_size=args.batch_size

    train(cfg, args)