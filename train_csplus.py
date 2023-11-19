import os
import io
import gzip
import random
import re
import sys
import argparse
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(30)
from torch.utils.data import Dataset, DataLoader, Subset
from networks import Unet, Unet2, rms, IFt, Ft
from utils import csplus_single, csplus_multi
from data import CMRDataset, PairDataset

import pytorch_lightning as pl
from torchmetrics import UniversalImageQualityIndex
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm import tqdm
from collections import OrderedDict, Counter
from skimage.metrics import peak_signal_noise_ratio

import matplotlib

nm = lambda x: (x - x.mean(dim=(-2,-1), keepdim=True)) / (x.std(dim=(-2,-1), keepdim=True) + 1e-9)
tv_fun = lambda I : (-I.roll(0, -1) + I.roll(1, -1)).norm(p=1) +\
                    (-I.roll(0, -2) + I.roll(1, -2)).norm(p=1)


def verbose_image(img_gt, img_recon):
    cmap = matplotlib.colormaps['pink']
    image_total = torch.cat([
        (img_recon / img_recon.max()), 
        (img_gt / img_gt.max())
    ], dim=0)
    image_total -= image_total.min()
    image_total /= image_total.max()
    return np.moveaxis(cmap(image_total.cpu().detach()), 2, 0)


def ssimloss(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: torch.Tensor,
    reduced: bool = True,
    win_size: int = 7, k1: float = 0.01, k2: float = 0.03
):
    w = torch.ones(1, 1, win_size, win_size, device=X.device) / win_size**2
    NP = win_size**2
    cov_norm = NP / (NP - 1)
    data_range = data_range[:, None, None, None]
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)  # typing: ignore
    uy = F.conv2d(Y, w)  #
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux**2 + uy**2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    if reduced:
        return 1 - S.mean()
    else:
        return 1 - S

class TikhonovReg(nn.Module):
    def __init__(self, size, reg) -> None:
        super().__init__()
        _R = reg * torch.eye(size)
        _R += torch.randn_like(_R) * reg * 1e-3
        self.R = torch.nn.Parameter(data=_R)
    def forward(self, x):
        return x + self.R

class CSplus(pl.LightningModule):

    def __init__(self, oname: str, lr: float, nchans: int, npool: int, isMapping: bool, niter: int, alr: float, aeps: float, reg: float, kscale: int, ksize: int):
        super().__init__()
        self.oname = oname
        self.lr = lr
        self.nchans = nchans
        self.npool = npool
        self.niter = niter
        self.alr = alr
        self.aeps = aeps
        self.isMapping = isMapping
        self.reg = reg
        self.kscale = kscale
        self.unet = nn.ModuleList([Unet(1, 1, nchans, npool) for _ in range(self.niter)])
        self.tikhonov = TikhonovReg(ksize, reg)
        self.automatic_optimization = False
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        # load batch [T x N x H x W]
        (_ks, _ks_gt, mask) = batch[0]
        for nslice in np.random.permutation(range(_ks.shape[1])):
            # [1 x N x H x W] -> [N x 1 x H x W]
            if self.isMapping:
                t = random.choice(range(_ks.shape[0]))
                ks = _ks[t:t+1,nslice:nslice+1].transpose(0,1)
                ks_gt = _ks_gt[t:t+1,nslice:nslice+1].transpose(0,1)
            else:
                ks = _ks[:1,nslice:nslice+1].transpose(0,1)
                ks_gt = _ks_gt[:1,nslice:nslice+1].transpose(0,1)
            # ground truth image
            img_gt = rms(IFt(ks_gt))
            # reconstructed image
            ks_tilda = csplus_multi(
                ks=ks / ks.norm(p=2, dim=(-2,-1), keepdim=True), 
                mask=mask, 
                func=lambda x,i: (x * self.unet[i](x).sigmoid()).norm(p=1),
                reg_func=lambda x: self.tikhonov(x),
                niter=self.niter, kscale=self.kscale, lr=self.alr, eps=self.aeps, train=True
            ) * ks.norm(p=2, dim=(-2,-1), keepdim=True)
            img_tilda = rms(IFt(ks_tilda))
            # L1 loss of images
            loss = (img_tilda - img_gt).norm(p=1)
            loss.backward()
            opt.step()
            opt.zero_grad()
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # load batch [T x N x H x W]
        (ks, ks_gt, mask) = batch[0]
        # [T x N x H x W] -> [N x 1 x H x W]
        if self.isMapping:
            t = ks.shape[0] // 2
            ks = ks[t:t+1].transpose(0,1)
            ks_gt = ks_gt[t:t+1].transpose(0,1)
        else:
            ks = ks[:1].transpose(0,1)
            ks_gt = ks_gt[:1].transpose(0,1)
        # ground truth image
        img_gt = rms(IFt(ks_gt))
        # reconstructed image
        with torch.enable_grad():
            ks_tilda = csplus_multi(
                ks=ks / ks.norm(p=2, dim=(-2,-1), keepdim=True), 
                mask=mask, 
                func=lambda x,i: (x * self.unet[i](x).sigmoid()).norm(p=1),
                reg_func=lambda x: self.tikhonov(x),
                niter=self.niter, kscale=self.kscale, lr=self.alr, eps=self.aeps, train=False
            ) * ks.norm(p=2, dim=(-2,-1), keepdim=True)
        img_tilda = rms(IFt(ks_tilda))
        # loss image
        loss = (img_tilda - img_gt).norm(p=1)
        # psnr of images
        im1 = (img_gt / img_gt.max()).data.cpu().numpy()
        im2 = (img_tilda / img_tilda.max()).data.cpu().numpy()
        psnr = peak_signal_noise_ratio(im1, im2, data_range=im1.max())
        self.logger.experiment.add_image(
            f'batch_idx{batch_idx}_{self.local_rank}', 
            verbose_image(img_gt[0,0], img_tilda[0,0]), 
            self.current_epoch
        )
        self.log('psnr', psnr, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        if self.oname == 'nadam':
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        elif self.oname == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.oname == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=9e-1, nesterov=True)
        else:
            raise ValueError
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cine', choices=['Cine', 'Mapping'], type=str,required=True)
    parser.add_argument('--subset', default='sax', choices=['sax', 'lax', 'T1', 'T2'], type=str,required=True)
    parser.add_argument('--x', default='10', choices=['04', '08', '10'], type=str,required=True)
    parser.add_argument('--c', default='SingleCoil', choices=['SingleCoil', 'MultiCoil'], type=str,required=True)
    parser.add_argument('--lr', default=1e-3, type=float, required=False)
    parser.add_argument('--oname', default='nadam', choices=['nadam', 'adam', 'sgd'], type=str, required=False)
    parser.add_argument('--nchans', default=32, type=int, required=False)
    parser.add_argument('--npool', default=4, type=int, required=False)
    parser.add_argument('--niter', default=5, type=int, required=False)
    parser.add_argument('--alr', default=1e-1, type=float, required=False)
    parser.add_argument('--aeps', default=1e0, type=float, required=False)
    parser.add_argument('--reg', default=1e-2, type=float, required=False)
    parser.add_argument('--path', type=str, required=False)
    parser.add_argument('--resume', type=str, required=False)
    parser.add_argument('--nnodes', type=int, default=1, required=False)
    parser.add_argument('--kscale', type=int, default=1, required=False)
    args = parser.parse_args()
    # RNG
    torch.manual_seed(228)
    random.seed(228)
    np.random.seed(228)
    # load GT and Accelerated dataset
    dataset_hat = CMRDataset(f'/root/datasets/ChallengeData/{args.c}/{args.dataset}/TrainingSet/AccFactor{args.x}', args.subset, load_all=True if args.dataset == 'Mapping' else False)
    dataset_full = CMRDataset(f'/root/datasets/ChallengeData/{args.c}/{args.dataset}/TrainingSet/FullSample', args.subset, load_all=True if args.dataset == 'Mapping' else False)
    # combine datasets in one
    paired_dataset = PairDataset(dataset_full, dataset_hat, isCached=False)
    # determine kernel size
    _, _, mask = paired_dataset[0]
    seq = (mask.abs() > 1e-12)[:, 0].cpu().numpy().tolist()
    seq = ''.join('u' if v else 'l' for v in seq)
    cnt = Counter(len(v) for v in seq.split('u'))
    kh = max(list(cnt.keys())) + 2
    kw = kh * args.kscale
    # init Proximal Gradient reconstruction with Unet L1 reg
    model = CSplus(oname=args.oname, lr=args.lr, nchans=args.nchans, npool=args.npool, isMapping=True if args.dataset == 'Mapping' else False, niter=args.niter, alr=args.alr, aeps=args.aeps, reg=args.reg, kscale=args.kscale, ksize=kh*kw)
    if args.path is not None:
        print(f'Load model from {args.path}')
        checkpoint = torch.load(args.path, map_location='cpu')
        unet_state_dict = OrderedDict()
        tikhonov_state_dict = OrderedDict()
        for k, p in checkpoint['state_dict'].items():
            if k.startswith('unet'):
                unet_state_dict[k.split('unet.')[-1]] = p
            if k.startswith('tikhonov'):
                tikhonov_state_dict[k.split('tikhonov.')[-1]] = p
        model.unet.load_state_dict(unet_state_dict)
        if len(tikhonov_state_dict) != 0:
            model.tikhonov.load_state_dict(tikhonov_state_dict)
    else:
        print('No weight loaded')
    # train val split
    trainset, valset = torch.utils.data.random_split(
        paired_dataset, 
        [len(paired_dataset) - 12, 12], 
        generator=torch.Generator().manual_seed(228)
    )
    # init data loaders - one batch per gpu
    trainloader = DataLoader(trainset, batch_size=1, num_workers=1, shuffle=True, collate_fn=lambda x: x)
    valloader = DataLoader(valset, batch_size=1, num_workers=1, collate_fn=lambda x: x)
    # pl callbacks
    callbacks = [
        ModelCheckpoint(
            save_last=True, save_top_k=7,
            save_weights_only=False, 
            monitor='psnr', 
            mode='max',
            filename='{epoch}-{psnr:.5f}',
        ),
    ]
    trainer = pl.Trainer(
        devices=torch.cuda.device_count(),
        num_nodes=args.nnodes,
        logger=TensorBoardLogger(save_dir='./logs', name=f'{args.c}/{args.dataset}/{args.subset}/Acc{args.x}', default_hp_metric=False),
        max_epochs=2000,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    # train model
    trainer.fit(model, trainloader, valloader, ckpt_path=args.resume)

