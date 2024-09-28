# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:39:00 2019

@author: Xiangyu Rui
"""
# torch版本的Unet_P3D
import jittor as jt
import torch
from jittor import nn
# from jittor.autograd import Function as autoF
# from jittor.autograd import Variable
from jittor.nn import Parameter
# import torch.nn.functional as F
import numpy as np
from math import log, pi, sqrt
import methods as ms

 
############################################################################################
#-------------------------------定义一个用于优化的变量-----------------------------------------#
############################################################################################
class Para(nn.Module):
    def __init__(self, ii):
        super(Para,self).__init__()
        self.X = Parameter(ii.clone())
        
    def execute(self):
        return self.X

##################################################################################################
#----------------------------------- HWLRMF (L2损失)---------------------------------------------#
##################################################################################################
class HWLRMF(nn.Module):
    def __init__(self, Ite=20, r = 3):
        super(HWLRMF, self).__init__()
        self.Ite = Ite
        self.r = r
        
    def execute(self, x, W, x_g=0):
        #初始化
        Band,Cha,Hei,Wid = x.size()
        W = W.reshape(Band, Cha, Hei*Wid)
 
        
        x = x.reshape(Band, Cha, Hei*Wid)
        r = self.r
        L = x.clone()
        
        rho = 0.5*torch.mean(W)      #这个配W使用
        
        alpha = 1.05
        loss = []
        loss_F = []
        #开始循环
        for i in range(self.Ite):
            
            U,_,_ = torch.linalg.svd(torch.matmul(L, L.transpose(1,2)))
            U = U[:,:,:r]
            UV = torch.matmul(U, torch.matmul(U.transpose(1,2), L))
            
            loss.append(torch.mean((UV.view(Band, Cha ,Hei, Wid) - x_g)**2).detach().cpu().numpy())          
            loss_F.append(torch.mean(W*(UV - x)**2).detach().cpu().numpy())
            
          #  mu = torch.exp(-torch.log(1+W*rho))  # 增加数值稳定吧
          #  L = mu*x + (1-mu)*UV
            mu = rho*torch.exp(-torch.log(W + rho))
            L = (1-mu)*x + mu*UV
            
            rho = alpha*rho
        
        return UV.reshape(Band, Cha, Hei, Wid), np.array(loss_F), np.array(loss)
    
#############################################################################################
#----------------------------- -- HWNUCLEAR (ALM) (L2损失)----------------------------------#
#############################################################################################
class HWNUCLR(nn.Module):
    def __init__(self, Ite=15, lam=1.3, mu=0.02):
        super(HWNUCLR, self).__init__()
        self.Ite = Ite
        self.lam = lam
        self.mu = mu
        
    def execute(self, y, inW, x_g=0, mp = 1):
        #初始化
        Band,Cha,Hei,Wid = y.size()
        W = inW.reshape(Band, Cha, Hei*Wid)
        
        y = y.reshape(Band, Cha, Hei*Wid)
        L = y.clone()
        
        mu = 2.5/sqrt(Hei*Wid)/mp
       # lam = mu/(0.1*torch.mean(W, dim=(-2,-1), keepdim=True)) # lam = mu/(0.1*mean(W))
        lam = mu/0.1
        
       # mu = self.mu
       # lam = self.lam*mp
        alpha = 1.03
        G = 0
        loss = []
        #开始循环
        
        for i in range(self.Ite):
            Z = ms.thres_mat(L + G/mu, 1/mu)
            
            L = (lam*W*y + mu*Z - G)/(lam*W + mu)
            
            G = G+mu*(L - Z)
            mu = mu*alpha
            
           # loss.append(torch.mean((Z.view(Band, Cha ,Hei, Wid) - x_g)**2).detach().cpu().numpy())
            
        return Z.view(Band, Cha ,Hei, Wid)/mp #, np.array(loss)
    
#############################################################################################
#-------------------------------HW-TV(spatial) (ALM)----------------------------------------#
#############################################################################################  
class HWTV(nn.Module):
    def __init__(self, Ite=20, shape=(10,31,64,64), lam=0.1, mu1=0.1, mu2=0.1):
        '''
        Parameters
        ----------
        y: [N,C,H,W] noisy input
        inW: [N,C,H,W] 实际指W^2
        
        Returns
        -------
        pred: [N,C,H,W]

        '''
        super(HWTV, self).__init__()
        self.Ite = Ite
        N,C,Hei,Wid = shape
        
        Dh = torch.Tensor([1, -1]).unsqueeze(0).unsqueeze(0).repeat(N,C,1,1) # 水平差，列-列
        Dv = torch.Tensor([[1],[-1]]).unsqueeze(0).unsqueeze(0).repeat(N,C,1,1)
        FH = ms.p2o(Dh, (Hei, Wid))
        FV = ms.p2o(Dv, (Hei, Wid))
        self.demo = abs(FH)**2 + abs(FV)**2
        self.lam = lam
        self.mu1 = mu1
        self.mu2 = mu2
    
    def execute(self, Y, inW, x_g=0, mp = 1):
    
        G1 = 0
        G21 = 0
        G22 = 0
        
      #  mu1 = torch.mean(inW, dim=(1,2,3), keepdim=True)
      #  mu2 = torch.mean(inW, dim=(1,2,3), keepdim=True)
      
        mu1 = self.mu1
        mu2 = self.mu2
        lam = self.lam

        rho = 1.05
       # Z = torch.zeros(Y.shape).to(device)
        Z = Y.clone()
        X = Y.clone()
        demo = self.demo
        
        DhZ = ms.diff_h(Z)
        DvZ = ms.diff_v(Z)
        mse_loss = []
        for i in range(self.Ite):
            
            # Update U
            Uh = ms.Shrink(DhZ - G21/mu2, lam/mu2)
            Uv = ms.Shrink(DvZ - G22/mu2, lam/mu2)
            
            # Update Z
            Z = ms.TV_solver(X + G1/mu1, Uh+G21/mu2, Uv+G22/mu2, demo, mu1, mu2)
            DhZ = ms.diff_h(Z)
            DvZ = ms.diff_v(Z)
            
            # Update X
            X = (inW*Y + mu1*Z - G1)/(inW + mu1)
            
            # Update G1, G21, G22
            G1 = G1 + mu1*(X - Z)
            G21 = G21 + mu2*(Uh - DhZ)
            G22 = G22 + mu2*(Uv - DvZ)
            
            mu1 = rho*mu1  # 这里不能用 mu1 *= rho 代替，这样的inplace=True就把原来的所有
            mu2 = rho*mu2
            
           # mse_loss.append(torch.mean(torch.pow(Z - x_g,2)).item())
        
        return Z #, mse_loss

#############################################################################################
#-------------------------------HW-TV(spectral) (ALM)---------------------------------------#
#############################################################################################  
class HWTV_S(nn.Module):
    def __init__(self, Ite=20, shape=(10,64,64,31), lam=0.1, mu1=0.1, mu2=0.1):
        '''
        Parameters
        ----------
        y: [N,H,W,C] noisy input, 注意C在最后一个维度
        inW: [N,H,W,C] 实际指W^2
        
        Returns
        -------
        pred: [N,H,W,C]

        '''
        super(HWTV_S, self).__init__()
        self.Ite = Ite
        N,Hei,Wid,C = shape
        self.lam = lam
        self.mu1 = mu1
        self.mu2 = mu2
        
        Ds = torch.Tensor([1, -1]).unsqueeze(0).unsqueeze(0).repeat(N,Hei,1,1)
        FS = ms.p2o(Ds, (Wid, C))
        self.demo = abs(FS)**2
    
    def execute(self, Y, inW, x_g=0, mp = 1):
        # device = Y.device
        N,Hei,Wid,C = Y.shape
        G1 = 0
        G2 = 0
        
      #  mu1 = 5*torch.mean(inW, dim = (1,2,3), keepdim=True)
      #  mu2 = 5*torch.mean(inW, dim = (1,2,3), keepdim=True)
        mu1 = self.mu1
        mu2 = self.mu2        
        lam = self.lam

        rho = 1.05
      #  Z = torch.zeros(Y.shape).to(device)
        Z = Y.clone()
        X = Y.clone()

        # 提前计算
        DsZ = ms.diff_h(Z) 
        demo = self.demo
        mse_loss = []
        for i in range(self.Ite):           
            # Update U
            Us = ms.Shrink(DsZ - G2/mu2, lam/mu2)
            
            # Update Z
            Z = ms.TV_solver_single(X + G1/mu1, Us+G2/mu2, demo, mu1, mu2)
            DsZ = ms.diff_h(Z)

            # Update X
            X = (inW*Y + mu1*Z - G1)/(inW + mu1)
            
            # Update G1, G21, G22
            G1 = G1 + mu1*(X - Z)
            G2 = G2 + mu2*(Us - DsZ)
            
            mu1 = rho*mu1  # 这里不能用 mu1 *= rho 代替，这样的inplace=True就把原来的所有
            mu2 = rho*mu2
            
          #  mse_loss.append(torch.mean(torch.pow(x_g-Z, 2)).item())
        
        return Z #, mse_loss
    
#############################################################################################
#-------------------------------HW-TV(spatial) (ALM)----------------------------------------#
#############################################################################################  
class HWTV3D(nn.Module):
    def __init__(self, Ite=20, shape=(10,31,64,64), lam=0.1, mu1=0.1, mu2=0.1):
        '''
        Parameters
        ----------
        y: [N,C,H,W] noisy input
        inW: [N,C,H,W] 实际指W^2
        
        Returns
        -------
        pred: [N,C,H,W]

        '''
        super(HWTV3D, self).__init__()
        self.Ite = Ite
        
        N,C,Hei,Wid = shape
        Dh = torch.Tensor([1, -1]).unsqueeze(0).unsqueeze(0).repeat(N,C,1,1) # 水平差，列-列
        Dv = torch.Tensor([[1], [-1]]).unsqueeze(0).unsqueeze(0).repeat(N,C,1,1)
        Ds = torch.Tensor([[1], [-1]]).unsqueeze(0).unsqueeze(0).repeat(N,Hei,1,1) # N,H,C,W
        FH = ms.p2o(Dh, (Hei, Wid))
        FV = ms.p2o(Dv, (Hei, Wid))
        FS = ms.p2o(Ds, (C, Wid)).permute(0,2,1,3)
        self.demo = abs(FH)**2 + abs(FV)**2 + abs(FS)**2 
        self.lam = lam
        self.mu1 = mu1
        self.mu2 = mu2
    
    def execute(self, Y, inW, x_g=0, mp = 1):
        # device = Y.device

        G1 = 0
        G21 = 0
        G22 = 0
        G23 = 0
        
        mu1 = self.mu1
        mu2 = self.mu2
        lam = self.lam
        
        rho = 1.05
       # Z = torch.zeros(Y.shape).to(device)
        Z = Y.clone()
        X = Y.clone()
        demo = self.demo
        DhZ = ms.diff_h(Z)
        DvZ = ms.diff_v(Z)
        DsZ = ms.diff_s(Z)
        
        mse_loss = []
        for i in range(self.Ite):
                     
            # Update U
            Uh = ms.Shrink(DhZ - G21/mu2, lam/mu2)
            Uv = ms.Shrink(DvZ - G22/mu2, lam/mu2)
            Us = ms.Shrink(DsZ - G23/mu2, lam/mu2)
            
            # Update Z
            Z = ms.TV_solver3D(X + G1/mu1, Uh+G21/mu2, Uv+G22/mu2, Us+G23/mu2, demo, mu1, mu2)
            DhZ = ms.diff_h(Z)
            DvZ = ms.diff_v(Z)
            DsZ = ms.diff_s(Z)

            # Update X
            X = (inW*Y + mu1*Z - G1)/(inW + mu1)
            
            # Update G1, G21, G22
            G1 = G1 + mu1*(X - Z)
            G21 = G21 + mu2*(Uh - DhZ)
            G22 = G22 + mu2*(Uv - DvZ)
            G23 = G23 + mu2*(Us - DsZ)
            
            mu1 = rho*mu1  # 这里不能用 mu1 *= rho 代替，这样的inplace=True就把原来的所有
            mu2 = rho*mu2
            
           # mse_loss.append(torch.mean(torch.pow(Z - x_g,2)).item())
            
        return Z #, mse_loss
        
        
#############################################################################################
#-------------------------------PnP-Denoiser------------------------------------------------#
#############################################################################################    
class HWPnP(nn.Module):
    def __init__(self, path, Ite = 15):
        super(HWPnP, self).__init__()
        self.denoiser = DnDenoiser()
        cks = torch.load(path)
        self.denoiser.load_state_dict(cks['model_state_dict'])
        for param in self.denoiser.parameters():
            param.requires_grad = False
        
        self.Ite = Ite
    
    def execute(self, y, inW, x_g=0, mp=1):
        '''
        Parameters
        ----------
        y : [C,1,B,H,W]
        inW : [C,1,B,H,W]
        x_g : [C,1,B,H,W]
        mp : default=1.
        
        Returns
        -------
        
        '''
        Cha, _, Band, Hei, Wid = y.shape
        # device = y.device
        x = y.clone()
        rho = 0.5   # rho = 0.5*mean(W)
        lam = (0.2**2)*rho
        sigma = torch.tensor(sqrt(lam/rho))
        mse_list = []
        for i in range(self.Ite):
            z = (inW*y + rho*x)*torch.exp(-torch.log(inW + rho))
            x = self.denoiser(torch.cat([z, sigma.repeat(Cha,1,Band,Hei,Wid)], dim=1))
            rho *= 1.2
            sigma = torch.tensor(sqrt(lam/rho))
          #  mse_list.append(torch.mean(torch.pow(x-x_g, 2)).item())
            
        return x #, np.array(mse_list)
    
#############################################################################################
#---------------------------------HW-WNNM---------------------------------------------------#
############################################################################################# 
def diag_embed(x):
    a = torch.tensor([1, 2, 3, 4]) # 一维张量
    n = a.size(0) # 获取张量的长度
    eye = torch.ones((n, n)) # 创建一个n*n的单位矩阵
    b = torch.mul(a, eye) # 将输入的张量与单位矩阵相乘
    return b



def closed_wnnm(X, C):
    U,S,V = torch.linalg.svd(X, full_matrices = False)
    # 每个batch 的newS情况不一样
    c2 = (S + 1e-12)**2 - 4*C
    c1 = S - 1e-12
    newS = torch.where(c2>=0, (c1+torch.sqrt(c2))/2, torch.Tensor([0.]).type(dtype=X.dtype))
    print('newS:', newS)
    X = torch.matmul(torch.matmul(U, diag_embed(newS)), V.transpose(1,2))
    return X

class HWWNNM(nn.Module):
    def __init__(self, Ite=15, mu=0.02, C=2):
        super(HWWNNM, self).__init__()
        '''
        min ||W\odot (Y - X)||_F^2 + ||X||_{\eta,*}

        y: [Bc, numpa, ims, ims]
        '''
        self.Ite = Ite
        self.mu = mu
        self.C = C
        
    def execute(self, y, inW):
        #初始化
        Band,Cha,Hei,Wid = y.shape
        inW = inW.reshape(Band, Cha, Hei*Wid)  # 此时一般Cha > Hei*Wid
        
        y = y.reshape(Band, Cha, Hei*Wid)
        L = y.clone()
        
        mu = 2.5/sqrt(Cha)/255
        C = mu/0.1
        
        alpha = 1.03
        G = 0
        loss = []
        #开始循环
        
        for i in range(self.Ite):
            Z = closed_wnnm(L - G/mu, 2*C/mu)
            
            L = (inW*y + mu*Z + G)/(inW + mu)
            
            G = G+mu*(Z - L)
            mu = mu*alpha
            
        return Z.view(Band, Cha ,Hei, Wid)

class NLSS_WNNM(nn.Module):
    def __init__(self, ite = 20, patchsize = 5, numpatch = 150, imsize=41):
        super(NLSS_WNNM, self).__init__()
        '''
        带有non local self similarity

        x : [BC,1,H,W]  WMMN采用band by band的方式
            x的范围是[0,255]
        '''
        self.dn = HWWNNM(Ite=ite)
        self.ps = patchsize
        self.numpa = numpatch
        self.ims = imsize

        self.Hnum = imsize - patchsize + 1
        self.Wnum = imsize - patchsize + 1
        self.totalnum = self.Hnum*self.Wnum

    def execute(self, x, inW):
        Bc = x.shape[0]
        x = x.squeeze(1) # x的大小是[BC, H, W]
        inW = inW.squeeze(1)

        mat = [x[b, h:h+self.ps, w:w+self.ps] for b in range(Bc) for h in range(self.Hnum) for w in range(self.Wnum)]
        mat = torch.stack(mat) # mat 的大小是 [Bc*totalnum, ps, ps] 

        inWmat = [inW[b, h:h+self.ps, w:w+self.ps] for b in range(Bc) for h in range(self.Hnum) for w in range(self.Wnum)]
        inWmat = torch.stack(inWmat)

        center_p = mat[round(self.totalnum/2)::self.totalnum,:,:].repeat_interleave(self.totalnum, dim=0)

        dist = torch.mean((mat - center_p)**2, dim=(1,2))
        dist = dist.split(self.totalnum, dim=0)
        mat = mat.split(self.totalnum, dim=0)
        inWmat = inWmat.split(self.totalnum, dim=0) 

        img = torch.zeros_like(x)
        weight = torch.zeros_like(x)
        order_all = []
        for i in range(Bc):
            _, order = dist[i].sort(dim=0, descending=False)
            order = order[:self.numpa]
            order_all.append(order)
            mat[i] = mat[i][order,:,:]
            inWmat[i] = inWmat[i][order, :, :]
        mat = torch.stack(mat) # mat大小是[Bc, numpa, ims, ims]
        inWmat = torch.stack(inWmat) 

        mat = self.dn(mat, inWmat, mp=255)
        
        for i in range(Bc):
            for j in range(self.numpa):
                index = order_all[i][j]
                hindex, windex = (index+1)//self.Hnum, (index+1)%self.Hnum-1
                img[i,hindex:hindex+self.ps, windex:windex+self.ps] = img[i,hindex:hindex+self.ps, windex:windex+self.ps] + mat[i, j, :, :]
                weight[i,hindex:hindex+self.ps, windex:windex+self.ps] += 1
        
        weight[ weight == 0 ] = 1

        img = img/weight

        return img/255


#############################################################################################
#-------------------------------HW-NGmeet---------------------------------------------------#
#############################################################################################    
class NGmeet(nn.Module):
    def __init__(self, cl=5*sqrt(2), patsize=4, Iter=5, lamada=0.54, k_subspace=6):
        super(NGmeet, self).__init__()
        self.cl = cl
        self.patsize = patsize
        self.Iter = Iter
        self.lamada = lamada
        self.k_subspace = k_subspace
    
    def execute(self, Nimg, inW, nSig):
        delta = 2
        Batn, Band, Hei, Wid = Nimg.shape
        
        Nimg = Nimg.reshape(Batn, Band, -1)*255
        Eimg = Nimg.clone() # reshape会共享内存
        inW = inW.reshape(Batn, Band, -1)
        
        rho = 9*torch.mean(inW, dim =(-2,-1), keepdim=True)
        lam = rho*torch.exp(-torch.log(inW + rho))
            
        for i in range(self.Iter):
            k_sub = self.k_subspace + delta*i
            E,_,_ = torch.linalg.svd(torch.matmul(Eimg, Eimg.transpose(1,2)), full_matrices=False)
            E = E[:,:,:k_sub]
            Eimg = torch.matmul(E.transpose(2,1), Eimg).reshape(Batn, k_sub, Hei, Wid) # 此时为reduced image 
            Nimg1 = torch.matmul(E.transpose(2,1), Nimg).reshape(Batn, k_sub, Hei, Wid)
            
            s1 = self.lamada*torch.sqrt(torch.abs(nSig**2 - torch.mean((Eimg - Nimg1)**2)))
            sigr = s1 if i>0 else nSig*sqrt(self.k_subspace/Band)
            
            Eimg = ms.c2p(Eimg, self.patsize)
          #  _, ts1, ts2 = Eimg.shape
            mean_tE = torch.mean(Eimg, dim=(1,2), keepdim=True)
            Eimg = ms.WNNM(Eimg - mean_tE, self.cl, sigr) + mean_tE
            Eimg = ms.p2c(Eimg, self.patsize, k_sub, int(Hei/self.patsize), int(Wid/self.patsize))
            Eimg1 = torch.matmul(E, Eimg)
            
            Eimg = (1-lam)*Nimg + lam*Eimg1 
            
        return Eimg1.reshape(Batn, Band, Hei, Wid)/255
    
###############################################################################################
# ------------------------------------构造 HWnet ----------------------------------------------#   
###############################################################################################
def conv3x3x1(in_chn, out_chn, bias=True):
    layer = nn.Conv3d(in_chn, out_chn, kernel_size = (1,3,3), stride = 1, padding = (0,1,1), bias = bias)  # 注意padding也要分方向
    return layer

def conv1x1x3(in_chn, out_chn, bias = True):
    layer = nn.Conv3d(in_chn, out_chn, kernel_size=(3,1,1), stride = 1, padding = (1,0,0), bias = bias)
    return layer

class HWnet(nn.Module):
    def __init__(self, in_chn=1, out_chn=1, dep=5, num_filters = 64, bias = True):
        super(HWnet,self).__init__()
        self.conv1 = conv3x3x1(in_chn, num_filters, bias=bias)
        self.conv2 = conv1x1x3(num_filters, num_filters, bias=bias)
      
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3x1(num_filters, num_filters, bias=bias))
            mid_layer.append(nn.ReLU(inplace=True))
            mid_layer.append(conv1x1x3(num_filters, num_filters, bias = bias))
          #  mid_layer.append(nn.BatchNorm3d(num_filters))
            mid_layer.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = nn.Conv3d(num_filters, out_chn, kernel_size = (3,3,3), stride = 1, padding = 1, bias = bias)
      #  self.BN = nn.BatchNorm3d(1)
        
        #initialization...
        print('Initialization...')
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
                 
    def execute(self, x):
        x = self.conv1(x)
        x = nn.Relu()(x)
        x = self.conv2(x)
        x = nn.Relu()(x)
        x = self.mid_layer(x)
        x = self.conv_last(x)
        
        return x

class HWnet2D(nn.Module):
    def __init__(self, in_chn=1, out_chn=1, dep=5, num_filters = 64, bias = True):
        super(HWnet2D,self).__init__()
        self.conv1 = nn.Conv2d(in_chn, num_filters, kernel_size = 3, stride = 1, padding = 1, bias = bias)
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(nn.Conv2d(num_filters, num_filters, kernel_size = 3, stride = 1, padding = 1, bias = bias))
            mid_layer.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = nn.Conv2d(num_filters, out_chn, kernel_size = 3, stride = 1, padding = 1, bias = bias)
        
        #initialization...
        print('Initialization...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
                 
    def execute(self, x):
        x = self.conv1(x)
        x = nn.Relu()(x)
        x = self.mid_layer(x)
        x = self.conv_last(x)
        
        return x
    
###############################################################################################
# ------------------------------------构造 Denoiser ------------------------------------------#   
###############################################################################################
class ResDenoiser(nn.Module):
    def __init__(self, in_chn=32, out_chn=31, num_filters = 64, bias = False):
        super(ResDenoiser,self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_chn, num_filters, kernel_size=(3,3), stride=1, padding=1, bias=bias),
                                  nn.BatchNorm2d(num_filters),
                                  nn.ReLU(inplace=True))
        self.resm1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), stride=1, padding=1, bias=bias),
                                   nn.BatchNorm2d(num_filters),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), stride=1, padding=1, bias=bias),
                                   nn.BatchNorm2d(num_filters))
        self.resm2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), stride=1, padding=1, bias=bias),
                                   nn.BatchNorm2d(num_filters),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), stride=1, padding=1, bias=bias),
                                   nn.BatchNorm2d(num_filters))
        self.resm3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), stride=1, padding=1, bias=bias),
                                   nn.BatchNorm2d(num_filters),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), stride=1, padding=1, bias=bias),
                                   nn.BatchNorm2d(num_filters))
        self.resm4 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), stride=1, padding=1, bias=bias),
                                   nn.BatchNorm2d(num_filters),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), stride=1, padding=1, bias=bias),
                                   nn.BatchNorm2d(num_filters))
        self.tail = nn.Conv2d(num_filters, out_chn, kernel_size=(3,3), stride=1, padding=1, bias=bias)
        
        self.apply(self._weights_init)
        
    def _weights_init(self,m):
        if isinstance(m, nn.Conv2d):
           nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
           
    def execute(self, x):
        x1 = self.head(x)
        x2 = nn.Relu()(x1 + self.resm1(x1))
        x3 = nn.Relu()(x2 + self.resm2(x2))
        x4 = nn.Relu()(x3 + self.resm3(x3))
        x5 = nn.Relu()(x4 + self.resm4(x4))
        x6 = self.tail(x5)
        
        return x6
    
class DnDenoiser(nn.Module):
    def __init__(self, in_chn=2, out_chn=1, dep=4, num_filters = 64, bias = False):
        super(DnDenoiser,self).__init__()
        self.head = nn.Sequential(conv3x3x1(in_chn, num_filters, bias=bias),
                                  nn.ReLU(inplace=True),
                                  conv1x1x3(num_filters, num_filters, bias=bias),
                                  nn.ReLU(inplace=True))
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3x1(num_filters, num_filters, bias=bias))
            mid_layer.append(nn.ReLU(inplace=True))
            mid_layer.append(conv1x1x3(num_filters, num_filters, bias = bias))
            mid_layer.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = nn.Conv3d(num_filters, out_chn, kernel_size = (3,3,3), stride = 1, padding = 1, bias = bias)
        
        #initialization...
        print('Initialization...')
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                 
    def execute(self, x):
        x1 = self.head(x)
        x2 = self.mid_layer(x1)
        x3 = self.conv_last(x2)
        
        return x3
    
     

    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            
