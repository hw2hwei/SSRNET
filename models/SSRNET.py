import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class SSRNET(nn.Module):
    def __init__(self, 
                 arch,
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        """Load the pretrained ResNet and replace top fc layer."""
        super(SSRNET, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands
        self.weight = nn.Parameter(torch.tensor([0.5]))

        self.conv_fus = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spat = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spec = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )

    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        gap_bands = self.n_bands / (self.n_select_bands-1.0)
        for i in range(0, self.n_select_bands-1):
            x_lr[:, int(gap_bands*i), ::] = x_hr[:, i, ::]
        x_lr[:, int(self.n_bands-1), ::] = x_hr[:, self.n_select_bands-1, ::]

        return x_lr

    def spatial_edge(self, x):
        edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
        edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

        return edge1, edge2

    def spectral_edge(self, x):
        edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

        return edge

    def forward(self, x_lr, x_hr):
        x = self.lrhr_interpolate(x_lr, x_hr)
        x = self.conv_fus(x)

        if self.arch == 'SSRNET':
            x_spat = x + self.conv_spat(x)
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat) 

            x_spec = x_spat + self.conv_spec(x_spat)
            spec_edge = self.spectral_edge(x_spec) 

            x = x_spec

        elif self.arch == 'SpatRNET':
            x_spat = x + self.conv_spat(x)

            spat_edge1, spat_edge2 = self.spatial_edge(x_spat) 
            x_spec = x
            spec_edge = self.spectral_edge(x_spec)

        elif self.arch == 'SpecRNET':
            x_spat = x
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat) 

            x_spec = x + self.conv_spec(x)
            spec_edge = self.spectral_edge(x_spec) 

            x = x_spec

        return x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge


