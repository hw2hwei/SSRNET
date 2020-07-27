from torch import nn
from utils import *
import cv2
import pdb
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam


def validate(test_list, arch, model, epoch, n_epochs):
    test_ref, test_lr, test_hr = test_list
    model.eval()

    psnr = 0
    with torch.no_grad():
        # Set mini-batch dataset
        ref = to_var(test_ref).detach()
        lr = to_var(test_lr).detach()
        hr = to_var(test_hr).detach()
        if arch == 'SSRNet':
            out, _, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpat':
            _, out, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpec':
            _, _, out, _, _, _ = model(lr, hr)
        else:
            out, _, _, _, _, _ = model(lr, hr)

        ref = ref.detach().cpu().numpy()
        out = out.detach().cpu().numpy()

        rmse = calc_rmse(ref, out)
        psnr = calc_psnr(ref, out)
        ergas = calc_ergas(ref, out)
        sam = calc_sam(ref, out)

        with open('ConSSFCNN.txt', 'a') as f:
            f.write(str(epoch) + ',' + str(rmse) + ',' + str(psnr) + ',' + str(ergas) + ',' + str(sam) + ',' + '\n')

    return psnr