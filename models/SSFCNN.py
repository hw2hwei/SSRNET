import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class SSFCNN(nn.Module):
    def __init__(self, 
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        """Load the pretrained ResNet and replace top fc layer."""
        super(SSFCNN, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands

        self.lrhr_conv1 = nn.Sequential(
                  nn.Conv2d(n_bands+n_select_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.lrhr_conv2 = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.lrhr_conv3 = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=5, stride=1, padding=2),
                  nn.ReLU(),
                )

    def forward(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.lrhr_conv1(x) 
        x = self.lrhr_conv2(x) 
        x = self.lrhr_conv3(x) 
        
        return x, 0, 0, 0, 0, 0


class ConSSFCNN(nn.Module):
    def __init__(self, 
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        """Load the pretrained ResNet and replace top fc layer."""
        super(ConSSFCNN, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands

        self.lrhr_conv1 = nn.Sequential(
                  nn.Conv2d(n_bands+n_select_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.lrhr_conv2 = nn.Sequential(
                  nn.Conv2d(n_bands+n_select_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.lrhr_conv3 = nn.Sequential(
                  nn.Conv2d(n_bands+n_select_bands, n_bands, kernel_size=5, stride=1, padding=2),
                  nn.ReLU(),
                )

    def forward(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.lrhr_conv1(x) 
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.lrhr_conv2(x) 
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.lrhr_conv3(x) 
        
        return x, 0, 0, 0, 0, 0
