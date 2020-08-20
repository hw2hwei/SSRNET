import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class ResTFNet(nn.Module):
    def __init__(self, 
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        """Load the pretrained ResNet and replace top fc layer."""
        super(ResTFNet, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands

        self.lr_conv1 = nn.Sequential(
                  nn.Conv2d(n_bands, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.lr_conv2 = nn.Sequential(
                  nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.lr_down_conv = nn.Sequential(
                  nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),
                )
        self.hr_conv1 = nn.Sequential(
                  nn.Conv2d(n_select_bands, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.hr_conv2 = nn.Sequential(
                  nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.hr_down_conv = nn.Sequential(
                  nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),
                )

        self.fusion_conv1 = nn.Sequential(
                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.fusion_conv2 = nn.Sequential(
                  nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),            
                )
        self.fusion_conv3 = nn.Sequential(
                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.fusion_conv4 = nn.Sequential(
                  nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),                
                )

        self.recons_conv1 = nn.Sequential(
                  nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                )
        self.recons_conv2 = nn.Sequential(
                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.recons_conv3 = nn.Sequential(
                  nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),
                )
        self.recons_conv4 = nn.Sequential(
                  nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
                )
        self.recons_conv5 = nn.Sequential(
                  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.recons_conv6 = nn.Sequential(
                  nn.Conv2d(64, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )

    def forward(self, x_lr, x_hr):

        # feature extraction
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x_lr = self.lr_conv1(x_lr)
        x_lr_cat = self.lr_conv2(x_lr)
        x_lr = x_lr + x_lr_cat
        x_lr = self.lr_down_conv(x_lr)

        x_hr = self.hr_conv1(x_hr)
        x_hr_cat = self.hr_conv2(x_hr)
        x_hr = x_hr + x_hr_cat
        x_hr = self.hr_down_conv(x_hr)
        x = torch.cat((x_hr, x_lr), dim=1)
        
        # feature fusion
        x = x + self.fusion_conv1(x)
        x_fus_cat = x
        x = self.fusion_conv2(x)
        x = x + self.fusion_conv3(x)
        x = self.fusion_conv4(x)
        x = torch.cat((x_fus_cat, x), dim=1)


        # image reconstruction
        x = self.recons_conv1(x)
        x = x + self.recons_conv2(x)
        x = self.recons_conv3(x)
        x = torch.cat((x_lr_cat, x_hr_cat, x), dim=1)
        x = self.recons_conv4(x)

        x = x + self.recons_conv5(x)
        x = self.recons_conv6(x)

        return x, 0, 0, 0, 0, 0


class TFNet(nn.Module):
    def __init__(self, 
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        """Load the pretrained ResNet and replace top fc layer."""
        super(TFNet, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands

        self.lr_conv1 = nn.Sequential(
                  nn.Conv2d(n_bands, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.lr_conv2 = nn.Sequential(
                  nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.lr_down_conv = nn.Sequential(
                  nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),
                )
        self.hr_conv1 = nn.Sequential(
                  nn.Conv2d(n_select_bands, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.hr_conv2 = nn.Sequential(
                  nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.hr_down_conv = nn.Sequential(
                  nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),
                )

        self.fusion_conv1 = nn.Sequential(
                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.fusion_conv2 = nn.Sequential(
                  nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),            
                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),                
                )

        self.recons_conv1 = nn.Sequential(
                  nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
                  nn.PReLU(),
                )
        self.recons_conv2 = nn.Sequential(
                  nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                  nn.Conv2d(64, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )

    def forward(self, x_lr, x_hr):

        # feature extraction
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x_lr = self.lr_conv1(x_lr)        
        x_lr_cat = self.lr_conv2(x_lr)
        x_lr = self.lr_down_conv(x_lr_cat)

        x_hr = self.hr_conv1(x_hr)        
        x_hr_cat = self.hr_conv2(x_hr)
        x_hr = self.hr_down_conv(x_hr_cat)
        x = torch.cat((x_hr, x_lr), dim=1)
        
        # feature fusion
        x = self.fusion_conv1(x)
        x = torch.cat((x, self.fusion_conv2(x)), dim=1)

        # image reconstruction
        x = self.recons_conv1(x)
        x = torch.cat((x_lr_cat, x_hr_cat, x), dim=1)
        x = self.recons_conv2(x)

        return x, 0, 0, 0, 0, 0
