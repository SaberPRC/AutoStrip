import os
import sys

dir_test = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_test)

import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed

from network.basic import ResidualBlock, InputTransition, OutputTransition, DownTransition, UpTransition, BasicBlock


class ASNet(nn.Module):
    """
    TODO: Full implementation for attention-driven, neighboring-aware segmentation network (ASNet)
    """
    def __init__(self, in_channels, out_channels, norm=None):
        super().__init__()
        self.in_tr_s = InputTransition(in_channels, 16, norm=None)
        self.in_tr_b = InputTransition(in_channels, 16, norm=None)

        self.down_32_s = DownTransition(16, 1, norm=norm)
        self.down_32_b = DownTransition(16, 1, norm=norm)
        self.fusion_32 = BasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.down_64_s = DownTransition(32, 1, norm=norm)
        self.down_64_b = DownTransition(32, 1, norm=norm)
        self.fusion_64 = BasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.down_128_s = DownTransition(64, 2)
        self.down_128_b = DownTransition(64, 2)
        self.fusion_128 = BasicBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.down_256_s = DownTransition(128, 2)
        self.down_256_b = DownTransition(128, 2)
        self.fusion_256 = BasicBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.up_256_s = UpTransition(256, 256, 2)
        self.up_256_b = UpTransition(256, 256, 2)
        self.fusion_up_256 = BasicBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.up_128_s = UpTransition(256, 128, 2)
        self.up_128_b = UpTransition(256, 128, 2)
        self.fusion_up_128 = BasicBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.up_64_s = UpTransition(128, 64, 1)
        self.up_64_b = UpTransition(128, 64, 1)
        self.fusion_up_64 = BasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.up_32_s = UpTransition(64, 32, 1)
        self.up_32_b = UpTransition(64, 32, 1)
        self.fusion_up_32 = BasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.out_tr_s = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_b = OutputTransition(32, out_channels, 'softmax')

    def forward(self, x):
        B, C, W, H, D = x.shape
        B_s, C_s, W_s, H_s, D_s = B, C, W - 32, H - 32, D - 32

        x_s = x[:, :, 16:W - 16, 16:H - 16, 16:D - 16]
        x_b = F.interpolate(x, size=[W_s, H_s, D_s], mode='trilinear')

        out_16_s = self.in_tr_s(x_s)
        out_16_b = self.in_tr_b(x_b)

        out_32_s = self.down_32_s(out_16_s)
        out_32_b = self.down_32_b(out_16_b)

        out_32_s = torch.cat([out_32_s, F.interpolate(out_32_b[:, :, 7:57, 7:57, 7:57], size=[64, 64, 64], mode='trilinear')], dim=1)
        out_32_s = self.fusion_32(out_32_s)

        out_64_s = self.down_64_s(out_32_s)
        out_64_b = self.down_64_b(out_32_b)

        out_64_s = torch.cat([out_64_s, F.interpolate(out_64_b[:, :, 4:28, 4:28, 4:28], size=[32, 32, 32], mode='trilinear')], dim=1)
        out_64_s = self.fusion_64(out_64_s)

        out_128_s = self.down_128_s(out_64_s)
        out_128_b = self.down_128_b(out_64_b)

        out_128_s = torch.cat([out_128_s, F.interpolate(out_128_b[:, :, 2:14, 2:14, 2:14], size=[16, 16, 16], mode='trilinear')], dim=1)
        out_128_s = self.fusion_128(out_128_s)

        out_256_s = self.down_256_s(out_128_s)
        out_256_b = self.down_256_b(out_128_b)

        out_256_s = torch.cat([out_256_s, F.interpolate(out_256_b[:, :, 1:7, 1:7, 1:7], size=[8, 8, 8], mode='trilinear')], dim=1)
        out_256_s = self.fusion_256(out_256_s)

        out_s = self.up_256_s(out_256_s, out_128_s)
        out_b = self.up_256_b(out_256_b, out_128_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 2:14, 2:14, 2:14], size=[16, 16, 16], mode='trilinear')], dim=1)
        out_s = self.fusion_up_256(out_s)

        out_s = self.up_128_s(out_s, out_64_s)
        out_b = self.up_128_b(out_b, out_64_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 4:28, 4:28, 4:28], size=[32, 32, 32], mode='trilinear')], dim=1)
        out_s = self.fusion_up_128(out_s)

        out_s = self.up_64_s(out_s, out_32_s)
        out_b = self.up_64_b(out_b, out_32_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 7:57, 7:57, 7:57], size=[64, 64, 64], mode='trilinear')], dim=1)
        out_s = self.fusion_up_64(out_s)

        out_s = self.up_32_s(out_s, out_16_s)
        out_b = self.up_32_b(out_b, out_16_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 13:115, 13:115, 13:115], size=[128, 128, 128], mode='trilinear')], dim=1)
        out_s = self.fusion_up_32(out_s)

        out_s = self.out_tr_s(out_s)
        out_b = self.out_tr_b(out_b)

        return out_s, out_b

