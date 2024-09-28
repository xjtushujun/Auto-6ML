# -*- coding: utf-8 -*-
"""
DATA, loss
"""
import jittor as jt
import numpy as np
from skimage.util import random_noise
import random as prand
import torch as torch
from scipy.special import digamma, gammaln
from jittor import Function as autoF
from math import log, pi
import cv2 as cv2
from functools import partial
from jittor.dataset import Dataset
import scipy.io as sio
import logging
#############################################################################################
# --------------训练数据集的生成 --------------------------------------------------------------
#############################################################################################
import jittor as jt
import numpy as np


def gaussian_kernel2(H, W, B, scale):
    centerSpa1 = np.random.randint(1,H-1, size=B)
    centerSpa2 = np.random.randint(1,W-1, size=B)
    XX, YY = np.meshgrid(np.arange(W), np.arange(H))
    out = np.exp((-(np.expand_dims(XX,-1)-centerSpa1)**2-(np.expand_dims(YY, -1)-centerSpa2)**2)/(2*scale**2))
    return out  #返回的是已经阶段好的map

def add_noniid_gaussian(x, *scale):
    pch_size = x.shape
    if scale == ():
        scale = np.random.uniform(32/2,128/2,size = pch_size[2]) #256为基准
    else:
        scale = scale[0]
    sig_mi = 5/255
    sig_ma = 75/255

    p_sigma_ = gaussian_kernel2(pch_size[0], pch_size[1], pch_size[2], scale)  # 生成谱段非连续的map
    p_sigma_ = (p_sigma_ - p_sigma_.min())/(p_sigma_.max()-p_sigma_.min())
    p_sigma_ = sig_mi + p_sigma_*(sig_ma - sig_mi)
    noise = np.random.randn(pch_size[0], pch_size[1], pch_size[2]) * p_sigma_
    x = x+ noise
    return x, p_sigma_

def add_iid_gaussian1(x, *sig):  # 所有iid
    if sig == ():
       sig = prand.uniform(10/255,70/255)
    else:
       sig = sig[0]
    s = x.shape
    x = x + np.random.randn(s[0],s[1],s[2])*sig
    return x, np.ones(s)*sig
 
def add_iid_gaussian2(x):       # 谱段iid
    s = x.shape
    sig = np.random.rand(s[-1])*(60/255)+10/255
    x = x+ np.random.randn(s[0], s[1], s[2])*sig
    return x, sig*np.ones(s)

def add_impulse(x,bn):
    B = x.shape[-1]
  #  ratio = prand.uniform(0.01,0.15)
    x,_ = add_iid_gaussian2(x)
    band = prand.sample(range(B), bn)
    ratio = np.random.uniform(0.1,0.5,size=bn)
    for i in range(bn):
        x[:,:,band[i]] = random_noise(x[:,:,band[i]], mode = 's&p', clip = False, amount = ratio[i])
    
    return x, band, ratio

def add_stripe(x, bn):
    N = x.shape[-2]
    B = x.shape[-1]
    x,_ = add_iid_gaussian2(x)
    band = prand.sample(range(B), bn)
    stripn = np.random.randint(int(N*0.05),int(N*0.2),size = bn)
    for i in range(bn):
        loc = prand.sample(range(N), stripn[i])
        stripes = np.random.rand(stripn[i])*0.5 - 0.25
        x[:,loc, band[i]] = x[:,loc, band[i]] - stripes
        
    return x, band, stripn

def add_deadline(x, bn):
    N = x.shape[-2]
    B = x.shape[-1]
    x,_ = add_iid_gaussian2(x)
    band = prand.sample(range(B), bn)
    dn = np.random.randint(int(N*0.05),int(N*0.2),size = bn)
    for i in range(bn):
        loc = prand.sample(range(N), dn[i])
        x[:,loc, band[i]] = 0
        
    return x, band, dn

ndict = {'iid1':add_iid_gaussian1,
         'iid2':add_iid_gaussian2,
         'non':add_noniid_gaussian,
         'impluse':partial(add_impulse, bn = 10),
         'stripe':partial(add_stripe, bn = 10),
         'deadline':partial(add_deadline, bn=10)}
nname = ['iid1','iid2','non','stripe', 'impluse', 'deadline']
        
def create_patch(im_mat, num_patch, nlist, r=6, bch_size=[64,31]):
    p_input = []
    p_label = []
    p_sigmap = []
    
    for i in range(num_patch):
        
        idx = np.random.randint(len(im_mat))
        size = im_mat[idx].shape
        
        px = np.random.randint(size[0] - bch_size[0])
        py = np.random.randint(size[1] - bch_size[0])
       # pz = np.random.randint(size[2] - pch_size[1])
        
        p_label_ = np.float32(im_mat[idx][px:px+bch_size[0], py:py+bch_size[0], :])
       # p_label_ = np.float32(im_mat[idx][px:px+pch_size[0], py:py+pch_size[0], pz:pz+pch_size[1]])
        rotAngle = np.random.randint(0,4)
        vFlip = np.random.randint(0,2)
        p_label_ = np.rot90(p_label_, rotAngle)
        if vFlip:
            p_label_ = p_label_[:,::-1,:] 
       # if prand.random() < 0.5:  # spectral shuffle
       #     p_label_ = p_label_[:,:,::-1]
         
        ntype = prand.sample(nlist, 1)[0]
        p_n= ndict[nname[ntype]](p_label_)
        if ntype in [0,1,2]:
            p_input_, sigmap = p_n[0], p_n[1]
        else:
            p_input_ = p_n[0]
            sigmap = sigma_estimate(p_input_, p_label_, 2*r+1, r)
                
        
        p_label.append(p_label_)
        p_input.append(p_input_)
        p_sigmap.append(sigmap)
    
    p_label = torch.from_numpy(np.array(p_label)).permute(0,3,1,2).type(torch.float32)
    p_input = torch.from_numpy(np.array(p_input)).permute(0,3,1,2).type(torch.float32)
    p_sigmap = torch.from_numpy(np.array(p_sigmap)).permute(0,3,1,2).type(torch.float32)  #python3.7 vision will give a mixed memory warning if using transpose or permute.
                                                                                          
    
    return p_input, p_label, p_sigmap

class Train_builder1(Dataset):
    def __init__(self, im_mat_list, num_patch, nlist):
        super(Train_builder1, self).__init__()
        self.num_patch = num_patch
        self.im_mat_list = im_mat_list
        self.ndict = {'iid1':add_iid_gaussian1,
                      'iid2':add_iid_gaussian2,
                      'non':add_noniid_gaussian,
                      'impluse':partial(add_impulse, bn = 10),
                      'stripe':partial(add_stripe, bn = 10),
                      'deadline':partial(add_deadline, bn=10)}
        self.nname = ['iid1','iid2','non','stripe', 'impluse', 'deadline']
        self.nlist = nlist
        
    def __len__(self):
        return self.num_patch
    
    def __getitem__(self, index):
        im_label = sio.loadmat(self.im_mat_list[index])['patch']
       # ntype = np.random.randint(0,6)
        ntype = prand.sample(self.nlist, 1)[0]
        tinput = self.ndict[self.nname[ntype]](im_label)
        im_input = tinput[0]
            
        im_label = torch.from_numpy(np.transpose(im_label.copy(), (2,0,1))).type(torch.float32)  # 这里地方要加上.copy()
        im_input = torch.from_numpy(np.transpose(im_input.copy(), (2,0,1))).type(torch.float32)

        return im_input, im_label
    
class Train_builder_sidd(Dataset):
    def __init__(self, im_mat_list, num_patch, nlist):
        super(Train_builder1, self).__init__()
        self.num_patch = num_patch
        self.im_mat_list = im_mat_list
        self.ndict = {'iid1':add_iid_gaussian1,
                      'iid2':add_iid_gaussian2,
                      'non':add_noniid_gaussian,
                      'impluse':partial(add_impulse, bn = 10),
                      'stripe':partial(add_stripe, bn = 10),
                      'deadline':partial(add_deadline, bn=10)}
        self.nname = ['iid1','iid2','non','stripe', 'impluse', 'deadline']
        self.nlist = nlist
        
    def __len__(self):
        return self.num_patch
    
    def __getitem__(self, index):
        im_label = sio.loadmat(self.im_mat_list[index])['patch']
       # ntype = np.random.randint(0,6)
        ntype = prand.sample(self.nlist, 1)[0]
        tinput = self.ndict[self.nname[ntype]](im_label)
        im_input = tinput[0]
            
        im_label = torch.from_numpy(np.transpose(im_label.copy(), (2,0,1))).type(torch.float32)  # 这里地方要加上.copy()
        im_input = torch.from_numpy(np.transpose(im_input.copy(), (2,0,1))).type(torch.float32)

        return im_input, im_label

def sigma_estimate(im_noisy, im_gt, win, sigma_spatial):
    noise2 = (im_noisy-im_gt)**2
    sigma2_map_est = cv2.GaussianBlur(noise2, (win, win), sigma_spatial)
    sigma2_map_est = sigma2_map_est.astype(np.float32)
    sigma2_map_est = np.sqrt(np.where(sigma2_map_est<1e-10, 1e-10, sigma2_map_est))
    return sigma2_map_est

def sta(img, mode):
    img = np.float32(img)
    if mode == 'all':
        ma = np.max(img)
        mi = np.min(img)
     #   return (img - mi)/(ma - mi)
        img = (img - mi)/(ma - mi)
        return img
    elif mode == 'pb':
        ma = np.max(img, axis=(0,1))
        mi = np.min(img, axis=(0,1))
        img = (img - mi)/(ma - mi)
        return img
        
    else:
        print('Undefined Mode!')
        return img

def logger_info(logger_name: str, log_path: str = 'default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s',
                                      datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)
#########################################################################################################################################
# -------------------------------------- 定义Loss ----------------------------------------------------------------------------------------
#########################################################################################################################################  
     
class Log_gamma(autoF):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input.detach().numpy()
        out = gammaln(input_np)
        out = out.type(dtype=input.dtype)
        
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.digamma(input) * grad_output
        
        return grad_input

log_gamma = Log_gamma.apply

def loss_repara(im_input, im_gt, pred_mu, m2, alpha, beta, rsam, ep0, sigmap, r=3):
    pred_mu.clamp_(min = log(1e-10), max=log(1e10))
    sigmap = sigmap**2
    p = 2*r+1
    
    log_beta = torch.log(beta)
    alpha_div_beta = torch.exp(torch.log(alpha) - log_beta)
    
    lh = 0.5*log(2*pi) + 0.5*torch.mean(log_beta - torch.digamma(alpha)) + 0.5*torch.mean(m2*alpha_div_beta)+ 0.5*torch.mean(rsam*(im_input - pred_mu)**2)
    
    kl_z_sig_sig = torch.mean( (alpha - p**2/2 -1)*torch.digamma(alpha) + gammaln(p**2/2+1) - log_gamma(alpha) + \
                              (p**2/2+1)*(log_beta - torch.log(p**2*sigmap/2)) + alpha_div_beta*0.5*p**2*sigmap - alpha)
    
    t_m = (pred_mu - im_gt)**2
    kl_z_sig_z = 0.5*torch.mean(t_m/ep0 + m2/ep0 - log(m2/ep0) -1)
    
    loss = lh + kl_z_sig_sig + kl_z_sig_z
    
    mse = torch.mean(t_m)
    
    return loss, lh, kl_z_sig_z, kl_z_sig_sig, mse

def loss_repara_laplace(im_input, im_gt, pred_mu, b, alpha, beta, rsam, lam, sigmap, p = 7):
    pred_mu.clamp_(min = log(1e-10), max=log(1e10))
    sigmap = sigmap**2
    
    log_beta = torch.log(beta)
    alpha_div_beta = torch.exp(torch.log(alpha) - log_beta)
    
    lh = 0.5*log(2*pi) + 0.5*torch.mean(log_beta - torch.digamma(alpha)) + 0.5*torch.mean((2*b**2)*alpha_div_beta)+ 0.5*torch.mean(rsam*(im_input - pred_mu)**2)
    
    kl_z_sig_sig = torch.mean((alpha - p**2/2 - 1)*torch.digamma(alpha) + gammaln(p**2/2+1) - log_gamma(alpha) + \
                           (p**2/2+1)*(log_beta - torch.log(p**2*sigmap/2)) + alpha_div_beta*0.5*p**2*sigmap - alpha)
    
    t_m = torch.abs(pred_mu - im_gt)
   # kl_z_sig_z = torch.mean(t_m)/lam + b/lam*torch.mean(torch.exp(-t_m/b)) - log(b/lam) -1  
    
    kl_z_sig_z = torch.mean(t_m)/lam + b/lam - log(b/lam) - 1 #上界
    
    loss = lh + kl_z_sig_sig + kl_z_sig_z
    
    mse = torch.mean((pred_mu - im_gt)**2)
    
    return loss, lh, kl_z_sig_z, kl_z_sig_sig, mse

def loss_klsig(alpha, beta, sigmap, p = 7):
    sigmap = sigmap**2
    
    log_beta = torch.log(beta)
    alpha_div_beta = torch.exp(torch.log(alpha) - log_beta)
    
    kl_z_sig_sig = torch.mean((alpha - p**2/2 - 1)*torch.digamma(alpha) + gammaln(p**2/2+1) - log_gamma(alpha) + \
                           (p**2/2+1)*(log_beta - torch.log(p**2*sigmap/2)) + alpha_div_beta*0.5*p**2*sigmap - alpha)
    
    return  kl_z_sig_sig
    
        
        

