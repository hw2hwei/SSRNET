import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class SpecCNN(nn.Module):
    def __init__(self, 
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        """Load the pretrained ResNet and replace top fc layer."""
        super(SpecCNN, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands

        self.lrhr_conv1 = nn.Sequential(
                  nn.Conv2d(n_select_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.lrhr_conv2 = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        
    def forward(self, x_lr, x_hr):
        x = x_hr
        x = self.lrhr_conv1(x)
        x = self.lrhr_conv2(x) 
        
        return x, 0, 0, 0, 0, 0


class SpatCNN(nn.Module):
    def __init__(self, 
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        """Load the pretrained ResNet and replace top fc layer."""
        super(SpatCNN, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands

        self.lrhr_conv1 = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.lrhr_conv2 = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        
    def forward(self, x_lr, x_hr):
        x = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x = self.lrhr_conv1(x)
        x = self.lrhr_conv2(x) 
        
        return x, 0, 0, 0, 0, 0


