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


args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print (args)


def main():
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Botswana':
      args.n_bands = 145
    elif args.dataset == 'KSC':
      args.n_bands = 176
    elif args.dataset == 'Urban':
      args.n_bands = 162
    elif args.dataset == 'IndianP':
      args.n_bands = 200
    elif args.dataset == 'Washington':
      args.n_bands = 191

    # Custom dataloader
    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)

    # Build the models
    if args.arch == 'SSFCNN':
      model = SSFCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands)
    elif args.arch == 'ConSSFCNN':
      model = ConSSFCNN(args.scale_ratio, 
                        args.n_select_bands, 
                        args.n_bands)
    elif args.arch == 'TFNet':
      model = TFNet(args.scale_ratio, 
                    args.n_select_bands, 
                    args.n_bands)
    elif args.arch == 'ResTFNet':
      model = ResTFNet(args.scale_ratio, 
                       args.n_select_bands, 
                       args.n_bands)
    elif args.arch == 'MSDCNN':
      model = MSDCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands)
    elif args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
      model = SSRNET(args.arch,
                     args.scale_ratio,
                     args.n_select_bands, 
                     args.n_bands)
    elif args.arch == 'SpatCNN':
      model = SpatCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands)
    elif args.arch == 'SpecCNN':
      model = SpecCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands)

    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))


    test_ref, test_lr, test_hr = test_list
    model.eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()

    begin_time = time()
    if args.arch == 'SSRNET':
        out, _, _, _, _, _ = model(lr, hr)
    elif args.arch == 'SpatRNET':
        _, out, _, _, _, _ = model(lr, hr)
    elif args.arch == 'SpecRNET':
        _, _, out, _, _, _ = model(lr, hr)
    else:
        out, _, _, _, _, _ = model(lr, hr)
    end_time = time()
    run_time = (end_time-begin_time)*1000

    print ()
    print ()
    print ('Dataset:   {}'.format(args.dataset))
    print ('Arch:   {}'.format(args.arch))
    print ('ModelSize(M):   {}'.format(np.around(os.path.getsize(model_path)//1024/1024.0, decimals=2)))
    print ('Time(Ms):   {}'.format(np.around(run_time, decimals=2)))
    print ()

    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    
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

    lr = np.squeeze(test_lr.detach().cpu().numpy())
    lr_red = lr[red, :, :][:, :, np.newaxis]
    lr_green = lr[green, :, :][:, :, np.newaxis]
    lr_blue = lr[blue, :, :][:, :, np.newaxis]
    lr = np.concatenate((lr_blue, lr_green, lr_red), axis=2)
    lr = 255*(lr-np.min(lr))/(np.max(lr)-np.min(lr))
    lr = cv2.resize(lr, (out.shape[2], out.shape[3]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('./figs/{}_lr.jpg'.format(args.dataset), lr)

    out = np.squeeze(out)
    out_red = out[red, :, :][:, :, np.newaxis]
    out_green = out[green, :, :][:, :, np.newaxis]
    out_blue = out[blue, :, :][:, :, np.newaxis]
    out = np.concatenate((out_blue, out_green, out_red), axis=2)
    out = 255*(out-np.min(out))/(np.max(out)-np.min(out))
    cv2.imwrite('./figs/{}_{}_out.jpg'.format(args.dataset, args.arch), out)

    ref = np.squeeze(ref)
    ref_red = ref[red, :, :][:, :, np.newaxis]
    ref_green = ref[green, :, :][:, :, np.newaxis]
    ref_blue = ref[blue, :, :][:, :, np.newaxis]
    ref = np.concatenate((ref_blue, ref_green, ref_red), axis=2)
    ref = 255*(ref-np.min(ref))/(np.max(ref)-np.min(ref))
    cv2.imwrite('./figs/{}_ref.jpg'.format(args.dataset), ref)

    lr_dif = np.uint8(1.5*np.abs((lr-ref)))
    lr_dif = cv2.cvtColor(lr_dif, cv2.COLOR_BGR2GRAY)
    lr_dif = cv2.applyColorMap(lr_dif, cv2.COLORMAP_JET)
    cv2.imwrite('./figs/{}_lr_dif.jpg'.format(args.dataset), lr_dif)

    out_dif = np.uint8(1.5*np.abs((out-ref)))
    out_dif = cv2.cvtColor(out_dif, cv2.COLOR_BGR2GRAY)
    out_dif = cv2.applyColorMap(out_dif, cv2.COLORMAP_JET)
    cv2.imwrite('./figs/{}_{}_out_dif.jpg'.format(args.dataset, args.arch), out_dif)


if __name__ == '__main__':
    main()
