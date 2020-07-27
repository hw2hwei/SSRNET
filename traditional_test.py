import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.TFNet import TFNet, ResTFNet
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MSDCNN import MSDCNN
from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam
from data_loader import build_datasets
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F
import cv2
from time import *
import scipy.io as scio

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print (args)


def main():
    # Custom dataloader

    # Set mini-batch dataset
    arch = 'CNMF'
    file_out = '{}_{}.mat'.format(args.dataset, arch)
    out = scio.loadmat(file_out)['I_CNMF']
    file_ref = '{}_ref.mat'.format(args.dataset)
    ref = scio.loadmat(file_ref)['S']
    out = np.swapaxes(out,0,2)
    ref = np.swapaxes(ref,0,2)

    out_max = np.max(out)
    out_min = np.min(out)
    out = 255*((out - out_min) / (out_max - out_min + 0.0))
    ref_max = np.max(ref)
    ref_min = np.min(ref)
    ref = 255*((ref - ref_min) / (ref_max - ref_min + 0.0))

    print ()
    print ()
    print ('Dataset:   {}'.format(args.dataset))
    print ('Arch:   {}'.format(arch))
    print ()
    
    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    print ('RMSE:   {:.4f};'.format(rmse))
    print ('PSNR:   {:.4f};'.format(psnr))
    print ('ERGAS:   {:.4f};'.format(ergas))
    print ('SAM:   {:.4f}.'.format(sam))

    # bands order
    if args.dataset == 'Botswana':
      red = 47
      green = 14
      blue = 3
    elif args.dataset == 'PaviaU' or args.dataset == 'Pavia':
      red = 66
      green = 28
      blue = 0
    elif args.dataset == 'KSC':
      red = 28
      green = 14
      blue = 3
    elif args.dataset == 'Urban':
      red = 25
      green = 10
      blue = 0
    elif args.dataset == 'Washington':
      red = 54
      green = 34
      blue = 10
    elif args.dataset == 'IndianP':
      red = 28
      green = 14
      blue = 3


    out = np.squeeze(out)
    out_red = out[red, :, :][:, :, np.newaxis]
    out_green = out[green, :, :][:, :, np.newaxis]
    out_blue = out[blue, :, :][:, :, np.newaxis]
    out = np.concatenate((out_blue, out_green, out_red), axis=2)
    out = 255*(out-np.min(out))/(np.max(out)-np.min(out))
    cv2.imwrite('./figs/{}_{}_out.jpg'.format(args.dataset, arch), out)

    ref = np.squeeze(ref)
    ref_red = ref[red, :, :][:, :, np.newaxis]
    ref_green = ref[green, :, :][:, :, np.newaxis]
    ref_blue = ref[blue, :, :][:, :, np.newaxis]
    ref = np.concatenate((ref_blue, ref_green, ref_red), axis=2)
    ref = 255*(ref-np.min(ref))/(np.max(ref)-np.min(ref))
    cv2.imwrite('./figs/{}_ref.jpg'.format(args.dataset), ref)

    out_dif = np.uint8(1.5*np.abs((out-ref)))
    out_dif = cv2.cvtColor(out_dif, cv2.COLOR_BGR2GRAY)
    out_dif = cv2.applyColorMap(out_dif, cv2.COLORMAP_JET)
    cv2.imwrite('./figs/{}_{}_out_dif.jpg'.format(args.dataset, arch), out_dif)


if __name__ == '__main__':
    main()
