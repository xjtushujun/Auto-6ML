# -*- coding: utf-8 -*-
"""
Created on Wed"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:54:29 2019

@author: Administrator
"""

import jittor as jt
import numpy as np
import os
import argparse
import jittor.nn as nn
import jittor.optim as optim
import scipy.io as sio
from net import HWnet
from net import HWNUCLR, HWTV, HWPnP, HWTV_S
import methods as ms
# from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as PSNR
import time as ti
import scipy.stats as sst
import glob
import lib as lib
import matplotlib.pyplot as plt
from math import log
# import jt.nn.functional as F
from os.path import join
import logging
import random as prand

parser = argparse.ArgumentParser(description = 'joint training')
parser.add_argument('--mode', dest = 'mode', default = 'train', help = 'train or test')
parser.add_argument('--ps', dest = 'patch_size', default = [64,31], type=list)
parser.add_argument('--bs', dest = 'batch_size', default = 20, type=int)
parser.add_argument('--save_path', dest = 'save_path', default = './cks/softmax_s0_dep5', type=str, help = 'pretrained models are saved here')
parser.add_argument('--saved_model',dest = 'saved_model', default ='', type=str, help='pre trained model')
parser.add_argument('--dataroot', dest = 'dataroot', type=str, default = '/media/jd/Model/Ruixy/DATA/HWnet_patches', help = 'data path')
parser.add_argument('--mn', dest='model_name', default='NUCLR_Complex', type=str)
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--log_dir', dest = 'log_dir', default = './log/' , type=str)
parser.add_argument('--lr', dest = 'learning_rate', default = 1e-3, type=float, help = 'learning rate')    #0.01for resnet
parser.add_argument('--epoch', dest = 'epoch', default = 10, type=int)
parser.add_argument('--gpu_en', default="1", help = 'GPU ids')
parser.add_argument('--ntype', dest='ntype', default=[4], type=list)
parser.add_argument('--fn', dest='fine_tune', action='store_true')
args = parser.parse_args()

jt.flags.use_cuda = 1

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_en
    print(args)
    patch_size = args.patch_size
    batch_size = args.batch_size

    ntype = [int(x) for x in args.ntype]
    print(ntype)
    fn = args.fine_tune
    
    model_name = args.model_name
    save_path = join(args.save_path, model_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    seed = args.seed
    np.random.seed(seed)
    prand.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    '''
    data_list_path =glob.glob('/home/iid/Ruixy/DATA/CAVE/*.mat')[0:20]
    print(len(data_list_path))
  #  data_list_path =glob.glob('F:/DATA/HY/CAVE/matfile/*.mat')[0:20]
    data = []
    for im in data_list_path:
        data.append(lib.sta(sio.loadmat(im)['A'], 'pb'))
    '''
    data_list_path =glob.glob(join(args.dataroot,'*.mat'))
    lent = len(data_list_path)
    Train_data = lib.Train_builder1(data_list_path, lent, ntype)
    Train_dataset = Train_data.set_attrs(batch_size=batch_size,shuffle=True) 
    Batch_group = len(Train_dataset)

    netS = HWnet(in_chn=1, out_chn=1, dep=5, bias=True)
    netS = netS
    netD_nuclr = HWNUCLR(Ite=15)
    
    optimizer = optim.Adam(netS.parameters(), lr=args.learning_rate)

    gamma = 0.8
    lr_scheduler = jt.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if args.saved_model:
        if fn:
            print('Load pre trained model for fine tune ' + args.saved_model)
            checkpoint = jt.load(join(args.saved_model))
            epoch_start = 0
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
            netS.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('Load pre trained model ' + args.saved_model)
            checkpoint = jt.load(join(save_path, args.saved_model))
            epoch_start = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            netS.load_state_dict(checkpoint['model_state_dict'])
    else:
        epoch_start = 0

    print('start training')
    
    param = [x for x in netS.parameters()]
    
    # build summary writer
    log_dir = join(save_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) 
    logger_name = 'train'
    lib.logger_info(logger_name, os.path.join(log_dir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    
    # upbound = jt.var([500])
    for ep in range(epoch_start, args.epoch):
        writer_loss = 0
        writer_l_nuclr = 0
        
        netS.train()   
        tic = ti.time()

        lr = optimizer.lr
        print('lr:',lr)
        if lr< 1e-6:
            print('Lowest learning rate warning! Break!')
            break
       # for ii in range(Batch_group):           
           # Binput, Blabel, _ = [x.type(torch.cuda.FloatTensor) for x in lib.create_patch(data, patch_size, nlist = [4], bch_size=batch_size)]
        for ii, t_p in enumerate(Train_dataset):           
            Binput, Blabel = [x.float() for x in t_p]   
            
            # optimizer.zero_grad()
            
            #单通道
            pred_map = netS(Binput.unsqueeze(1)).squeeze(1)
            ssb1, ssb2, ssb3, ssb4 = pred_map.shape
           # W = torch.minimum(torch.exp(pred_map), upbound) + 1e-4
            W = ms.my_softmax(pred_map) + 1e-4
            
            pred_nuclr = netD_nuclr(Binput*255, W, mp=255)
            
            loss_nuclr = nn.mse_loss(pred_nuclr, Blabel)
            
            loss = loss_nuclr 
            
            # loss.backward()
            optimizer.step(loss)

            writer_loss += loss.item()

            if (ii+1)%20 == 0:
                total_norm = 0
                for x in param:
                    total_norm += jt.mean(jt.abs(x.grad))
                total_norm /= len(param)
            
                # logger.info(f"Batch [{ii+1:4d}]/[{Batch_group:4d}][{ep+1:4d}] -- Loss: {loss.item():.3e} -- Grad: {total_norm.item():.3e}")
            
        lr_scheduler.step()    
            
        toc = ti.time()
        
        # logger.info(f"----- Epoch [{ep+1:4d}]/[{args.epoch:4d}] -- Loss: {writer_loss/ Batch_group:.3e} -- Time: {toc - tic:.4f}")
        print('#######################')

              
        #save model
        jt.save({
            'epoch': ep+1,
            'model_state_dict': netS.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, join(save_path, model_name +'_' + str(ep+1) + '.pkl'))
        ep +=1
        
    print('Finish training!')
    
        
            
                
            
            
        


















