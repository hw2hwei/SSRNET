import torch
from torch import nn
from utils import to_var, batch_ids2words
import random
import torch.nn.functional as F
import cv2


def spatial_edge(x):
    edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
    edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

    return edge1, edge2

def spectral_edge(x):
    edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

    return edge


def train(train_list, 
          image_size, 
          scale_ratio, 
          n_bands, 
          arch, 
          model, 
          optimizer, 
          criterion, 
          epoch, 
          n_epochs):
    train_ref, train_lr, train_hr = train_list

    h, w = train_ref.size(2), train_ref.size(3)
    h_str = random.randint(0, h-image_size-1)
    w_str = random.randint(0, w-image_size-1)

    train_lr = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]
    train_ref = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]
    train_lr = F.interpolate(train_ref, scale_factor=1/(scale_ratio*1.0))
    train_hr = train_hr[:, :, h_str:h_str+image_size, w_str:w_str+image_size]

    model.train()

    # Set mini-batch dataset
    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()
    image_ref = to_var(train_ref).detach()

    # Forward, Backward and Optimize
    optimizer.zero_grad()

    out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = model(image_lr, image_hr)
    ref_edge_spat1, ref_edge_spat2 = spatial_edge(image_ref)
    ref_edge_spec = spectral_edge(image_ref)

    if 'RNET' in arch:        
        loss_fus = criterion(out, image_ref)
        loss_spat = criterion(out_spat, image_ref)
        loss_spec = criterion(out_spec, image_ref)
        loss_spec_edge = criterion(edge_spec, ref_edge_spec)
        loss_spat_edge = 0.5*criterion(edge_spat1, ref_edge_spat1) + 0.5*criterion(edge_spat2, ref_edge_spat2)
        if arch == 'SpatRNET':
            loss = loss_spat + loss_spat_edge
        elif arch == 'SpecRNET':
            loss = loss_spec + loss_spec_edge
        elif arch == 'SSRNET':
            loss = loss_fus + loss_spat_edge + loss_spec_edge
    else:
        loss = criterion(out, image_ref)

    loss.backward()
    optimizer.step()

    # Print log info
    print('Epoch [%d/%d], Loss: %.4f'
          %(epoch, 
            n_epochs, 
            loss,
            ) 
         )
