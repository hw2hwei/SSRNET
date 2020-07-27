import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from googlenet import googlenet
from utils import batch_ids2words
import cv2


class MSDCNN(nn.Module):
    def __init__(self, 
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        """Load the pretrained ResNet and replace top fc layer."""
        super(MSDCNN, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands

        self.shallow_conv = nn.Sequential(
                  nn.Conv2d(n_bands+n_select_bands, 64, kernel_size=9, stride=1, padding=4),
                  nn.PReLU(),
                  nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
                  nn.PReLU(),
                  nn.Conv2d(32, n_bands, kernel_size=5, stride=1, padding=2),
                  nn.PReLU(),
                )
        self.deep_conv_a = nn.Sequential(
                  nn.Conv2d(n_bands+n_select_bands, 60, kernel_size=7, stride=1, padding=3),
                  nn.PReLU(),
                )
        self.deep_conv_b_3x3 = nn.Sequential(
                  nn.Conv2d(60, 20, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.deep_conv_b_5x5 = nn.Sequential(
                  nn.Conv2d(60, 20, kernel_size=5, stride=1, padding=2),
                  nn.PReLU(),
                )
        self.deep_conv_b_7x7 = nn.Sequential(
                  nn.Conv2d(60, 20, kernel_size=7, stride=1, padding=3),
                  nn.PReLU(),
                )
        self.deep_conv_c = nn.Sequential(
                  nn.Conv2d(60, 30, kernel_size=7, stride=1, padding=3),
                  nn.PReLU(),
                )
        self.deep_conv_d_3x3 = nn.Sequential(
                  nn.Conv2d(30, 10, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.deep_conv_d_5x5 = nn.Sequential(
                  nn.Conv2d(30, 10, kernel_size=5, stride=1, padding=2),
                  nn.PReLU(),
                )
        self.deep_conv_d_7x7 = nn.Sequential(
                  nn.Conv2d(30, 10, kernel_size=7, stride=1, padding=3),
                  nn.PReLU(),
                )
        self.deep_conv_e = nn.Sequential(
                  nn.Conv2d(30, n_bands, kernel_size=5, stride=1, padding=2),
                  nn.PReLU(),
                )


    def forward(self, x_lr, x_hr):

        # feature extraction
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x = torch.cat((x_hr, x_lr), dim=1)
        x_shallow = self.shallow_conv(x)

        x_deep = self.deep_conv_a(x)
        x_deep = x_deep + torch.cat((self.deep_conv_b_3x3(x_deep),
                                     self.deep_conv_b_5x5(x_deep),
                                     self.deep_conv_b_7x7(x_deep)),
                                    dim=1)
        x_deep = self.deep_conv_c(x_deep)
        x_deep = x_deep + torch.cat((self.deep_conv_d_3x3(x_deep),
                                     self.deep_conv_d_5x5(x_deep),
                                     self.deep_conv_d_7x7(x_deep)),
                                    dim=1)
        x_deep = self.deep_conv_e(x_deep)
        x = x_shallow + x_deep

        return x, 0, 0, 0, 0, 0
