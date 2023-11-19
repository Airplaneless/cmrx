import numpy as np
import torch
import torch.nn.functional as F
from networks import rms, Ft, IFt
from collections import Counter


def dice_loss(gt: torch.tensor, logits: torch.tensor, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[gt.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[gt.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, gt.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss

def getGRAPPA(ks, kh: int, kw: int, reg_func: callable):
    dh, dw = 1, 1  # stride
    k = ks[0, 0, :, ks.shape[3]//2 - 12:ks.shape[3]//2 + 12, :].clone()             # TODO: hardcoded size of calibration lines!
    ker = k.unfold(1, kh, dh).unfold(2, kw, dw).permute(1,2,0,3,4)
    Ac = ker.flatten(0,1).flatten(1,3)
    ncoils = ker.shape[-3]
    _, _S, G = torch.linalg.svd(Ac, full_matrices=False)
    G = G[:kh*kw]
    kkernels = G.reshape(-1, ncoils, kh, kw)
    W = torch.zeros(kh - 2, 1, ncoils, ncoils * 2 * kw, device=ks.device).cfloat()
    x = kkernels.clone()
    x = torch.cat([x[:, :, 0, :][:, :, None], x[:, :, -1, :][:, :, None]], dim=2)
    x = x.contiguous().flatten(1).T
    for i in range(kh - 2):
        for j in range(1):
            b = kkernels[:, :, i + 1, j].clone()
            b = b.contiguous().T
            W[i][j] = b @ (reg_func(x.conj().T @ x).inverse() @ x.conj().T)
    return W

def predictGRAPPA(ks, mask, W, kh: int, kw: int):
    dh, dw = 1, 1  # stride
    ncoils = ks.shape[-3]
    ks_grappa = ks[0,0].clone()
    seq = (mask.abs() > 1e-12)[:, 0].float()
    for i in range(ks_grappa.shape[1] - kh):
        if seq[i].item() == 1 and seq[i + kh - 1].item() == 1 and not (seq[i+1:i+kh] == 1).all():
            p = ks_grappa[:, i:i+kh, :]\
                .unfold(1, kh, dh)\
                .unfold(2, kw, dw)\
                .permute(1,2,0,3,4)\
                .reshape(-1, ncoils, kh, kw)
            p = torch.cat([
                p[:, :, 0, :][:, :, None], 
                p[:, :, -1, :][:, :, None]
            ], dim=2).flatten(1).T
            l = W @ p
            ks_grappa[:, i + 1:i + kh - 1, :l.shape[-1]] = l[:, 0, :, :].permute(1,0,2).clone()
    return ks_grappa[None]


def csplus_multi(ks, mask, func, reg_func, niter=5, kscale=1, lr=5e-4, eps=5e-2, beta1=0.9, beta2=0.999, l1coeff=5e0, train=False):
    # make grappa prediction
    assert ks.shape[1] == 1
    seq = (mask.abs() > 1e-12)[:, 0].cpu().numpy().tolist()
    seq = ''.join('u' if v else 'l' for v in seq)
    cnt = Counter(len(v) for v in seq.split('u'))
    kh = max(list(cnt.keys())) + 2
    kw = kh * kscale
    ks_grappa = torch.zeros_like(ks)
    for b in range(ks.shape[0]):
        W = getGRAPPA(ks[b:b+1], kh=kh, kw=kw, reg_func=reg_func)
        ks_grappa[b:b+1] = predictGRAPPA(ks[b:b+1], mask, W, kh=kh, kw=kw)
    _l = torch.nn.Parameter(data=torch.zeros_like(ks[:, :, :, (1 - mask).bool()]), requires_grad=True)
    _l_moment1 = torch.zeros_like(_l).clone().requires_grad_(True)
    _l_moment2 = torch.zeros_like(_l).clone().requires_grad_(True)
    for i in range(niter):
        __ks = torch.zeros_like(ks[:, :])
        __ks[:, :, :, mask.bool()] = ks[:, :, :, mask.bool()]
        __ks[:, :, :, (1 - mask).bool()] = _l + ks_grappa[:, :, :, (1 - mask).bool()]
        img_recon = rms(IFt(__ks))
        loss = l1coeff * func(img_recon, i)
        l_grad, = torch.autograd.grad(loss, [_l], create_graph=train)
        _l_moment1 = _l_moment1.mul(beta1).add(l_grad.mul(1 - beta1))
        _l_moment2 = _l_moment2.mul(beta2).add(l_grad.mul(l_grad.conj()).mul(1 - beta2))
        bias_correction1 = 1 - torch.pow(beta1, torch.tensor(i+1).to(_l.device))
        bias_correction2 = 1 - torch.pow(beta2, torch.tensor(i+1).to(_l.device))
        step_size = lr / bias_correction1
        step_size_neg = step_size.neg()
        bias_correction2_sqrt = bias_correction2.sqrt()
        denom = (_l_moment2.sqrt() / (bias_correction2_sqrt * step_size_neg)) + (eps / step_size_neg)
        _l = _l.add(_l_moment1.div(denom))
        _l.retain_grad()
    ks_tilda = torch.zeros_like(ks[:,:])
    ks_tilda[:, :, :, (1 - mask).bool()] = _l + ks_grappa[:, :, :, (1 - mask).bool()]
    ks_tilda[:, :, :, mask.bool()] = ks[:, :, :, mask.bool()]
    return ks_tilda


def csplus_single(ks, mask, func, niter=5, lr=5e-4, eps=5e-2, beta1=0.9, beta2=0.999, l1coeff=5e0, train=False):
    _l = torch.nn.Parameter(data=torch.zeros_like(ks[:, :, :, (1 - mask).bool()]), requires_grad=True)
    _l_moment1 = torch.zeros_like(_l).clone().requires_grad_(True)
    _l_moment2 = torch.zeros_like(_l).clone().requires_grad_(True)
    for i in range(niter):
        __ks = torch.zeros_like(ks[:, :])
        __ks[:, :, :, mask.bool()] = ks[:, :, :, mask.bool()]
        __ks[:, :, :, (1 - mask).bool()] = _l
        img_recon = rms(IFt(__ks))
        loss = l1coeff * func(img_recon, i)
        l_grad, = torch.autograd.grad(loss, [_l], create_graph=train)
        _l_moment1 = _l_moment1.mul(beta1).add(l_grad.mul(1 - beta1))
        _l_moment2 = _l_moment2.mul(beta2).add(l_grad.mul(l_grad.conj()).mul(1 - beta2))
        bias_correction1 = 1 - torch.pow(beta1, torch.tensor(i+1).to(_l.device))
        bias_correction2 = 1 - torch.pow(beta2, torch.tensor(i+1).to(_l.device))
        step_size = lr / bias_correction1
        step_size_neg = step_size.neg()
        bias_correction2_sqrt = bias_correction2.sqrt()
        denom = (_l_moment2.sqrt() / (bias_correction2_sqrt * step_size_neg)) + (eps / step_size_neg)
        _l = _l.add(_l_moment1.div(denom))
        _l.retain_grad()
    ks_tilda = torch.zeros_like(ks[:,:])
    ks_tilda[:, :, :, (1 - mask).bool()] = _l
    ks_tilda[:, :, :, mask.bool()] = ks[:, :, :, mask.bool()]
    return ks_tilda

nm = lambda x: (x - x.mean(dim=(-2,-1), keepdim=True)) / (x.std(dim=(-2,-1), keepdim=True) + 1e-9)

def calculate_dsc(pred,gt,tissue_index):
    pred_tissue  = (pred == tissue_index).astype(int)
    gt_tissue    = (gt == tissue_index).astype(int)
    return 2*(np.sum((pred_tissue + gt_tissue) == 2))/(np.sum(pred_tissue) + np.sum(gt_tissue)), np.sum(gt_tissue)

def calculate_weighted_dsc(pred,gt):
    scaling_multiplier = 0
    weighted_dsc       = 0
    for tissue_ in range(1,4):
        curr_dsc, n_tissue_pixels = calculate_dsc(pred,gt,tissue_)
        weighted_dsc += curr_dsc/n_tissue_pixels
        scaling_multiplier += 1/n_tissue_pixels

    weighted_dsc = weighted_dsc*(1/scaling_multiplier)
    return weighted_dsc
