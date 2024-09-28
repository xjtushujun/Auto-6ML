# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:11:28 2021

@author: Administrator
"""
import torch
from jittor import nn
# from torch.autograd import Function as autoF
# from torch.autograd import Variable
# import torch.nn.functional as F
import numpy as np
from math import log, pi, sqrt
from scipy.special import digamma, gammaln   
import numpy.fft as FFT

#########################################################################################
#-------------------------------- 一些自定义的函数 ---------------------------------------#
#########################################################################################
def Shrink(x, lam): 
    tx = (torch.abs(x) - lam).type(dtype=x.dtype)
    tx = torch.where(tx>0, tx, torch.Tensor([0.]).type(dtype=x.dtype))
    y = nn.sign(x)*tx
    return y


def diag_embed(x):
    a = torch.tensor([1, 2, 3, 4]) # 一维张量
    n = a.size(0) # 获取张量的长度
    eye = torch.ones((n, n)) # 创建一个n*n的单位矩阵
    b = torch.mul(a, eye) # 将输入的张量与单位矩阵相乘
    return b


def thres_mat(X, lam):
    U,S,_ = torch.linalg.svd(torch.matmul(X, X.transpose(1,2)), full_matrices = False)
    S = torch.sqrt(S)
    VT = torch.matmul(diag_embed(1/S), torch.matmul(U.transpose(1,2), X))
    S = (S - lam).type(dtype=X.dtype)
    S = torch.where(S>0, S, torch.Tensor([0.]).type(dtype=X.dtype))
    Y = torch.matmul(U,torch.matmul(diag_embed(S), VT))
    return Y

def thres_mat_clipr(X, lam, clip_r):
    U,S,_ = torch.linalg.svd(torch.matmul(X, X.transpose(1,2)), full_matrices = False)
    S = torch.sqrt(S[:,:clip_r])
    U = U[:,:,:clip_r]
    VT = torch.matmul(torch.diag_embed(1/S), torch.matmul(U.transpose(1,2), X))
    S = (S - lam).type(dtype=X.dtype)
    S = torch.where(S>0, S, torch.Tensor([0.]).type(dtype=X.dtype))
    Y = torch.matmul(U,torch.matmul(diag_embed(S), VT))
    return Y

def WNNM(Y, C, Nsig):
    U,S,V = torch.linalg.svd(Y, full_matrices=False) #此时输出为我们通常所理解的V转置
    _,_,PatNum = Y.shape
    TempC = C*sqrt(PatNum)*2*(Nsig**2)
    temp = nn.ReLU(inplace=True)((S + 1e-15)**2 - 4*TempC)
    tempS = ((S - 1e-15 + torch.sqrt(temp))/2).type(dtype=Y.dtype)
    SX = torch.where(temp>0, tempS, torch.Tensor([0.]).type(dtype=Y.dtype))
    X = torch.matmul(U, torch.matmul(diag_embed(SX), V)) #配合torch.linalg.svd
    return X 

def nuclear_norm(Y):
    _,S,_ = torch.linalg.svd(torch.matmul(Y, Y.transpose(1,2)), full_matrices=False)
    S = torch.sqrt(S)
    nuclear_norm = torch.sum(S)
    return nuclear_norm

def L1_norm(Y):
    L1_norm = torch.sum(torch.abs(Y))
    return L1_norm

def HTV(Y):
    _,_,h,w = Y.shape
    h_tv = torch.abs(Y[:,:,:h-1,:] - Y[:,:,1:,:])
    w_tv = torch.abs(Y[:,:,:,:w-1] - Y[:,:,:,1:])
    htv = torch.sum(h_tv) + torch.sum(w_tv)
    return htv

def p2c(patch, patsize, Band, n1, n2):
    Batn, _, _  = patch.shape
    patch = patch.reshape(Batn, Band, patsize, patsize, n1, n2).permute(0,1,4,2,5,3)
    cube = patch.reshape(Batn, Band, patsize*n1*patsize*n2)
    return cube
        
    
def c2p(cube, patsize):
    Batn, Band, Hei, Wid = cube.shape
    n1 = int(Hei/patsize)
    n2 = int(Wid/patsize)
    cube = cube.reshape(Batn, Band, n1, patsize, n2, patsize).permute(0,1,3,5,2,4)
    patch = cube.reshape(Batn, Band*patsize*patsize, n1*n2)
    return patch


def view_as_real(input):
    # check if input is a complex tensor
    # if not input.is_complex():
    #     raise TypeError("input should be a complex tensor")
    # get the shape of input
    shape = input.shape
    # create a new tensor with an extra dimension of size 2
    output = jt.zeros(shape + [2])
    # assign the real and imaginary parts to the output
    output[..., 0] = input.real()
    output[..., 1] = input.imag()
    return output


def p2o(psf, shape): # 可以把fft换成rfft，结果是不变的
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = FFT.rfft2(otf)
    otf = view_as_real(otf)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    otf = view_as_complex(otf)
    return otf

def TV_solver(f, Uh, Uv, demo, mu1, mu2):
    Uhc = diff_hc(Uh)
    Uvc = diff_vc(Uv)
    
    UP = FFT.rfft2(Uhc + Uvc + (mu1/mu2)*f, s=list(f.shape[-2:]))
    x = FFT.irfft2(UP/(demo + mu1/mu2), s=list(f.shape[-2:]))
    return x

def TV_solver_single(f, Uh, demo, mu1, mu2):
    Uhc = diff_hc(Uh)
    
    UP = FFT.rfft2(Uhc + (mu1/mu2)*f, s=list(f.shape[-2:]))
    x = FFT.irfft2(UP/(demo + mu1/mu2), s=list(f.shape[-2:]))
    return x

def TV_solver3D(f, Uh, Uv, Us, demo, mu1, mu2): 
    # FFT.rfft2使用的是矩阵精简形式，为了避免Us和Uh，Uv的维度不一致，这里计算UP时采用等式F(A).F(B)=F(A*B)的右边的计算形式
    Uhc = diff_hc(Uh)
    Uvc = diff_vc(Uv)
    Usc = diff_sc(Us)
    
    UP = FFT.rfftn(Uhc + Uvc + Usc + (mu1/mu2)*f, s=list(f.shape[-3:]), dim=(-3,-2,-1))
    x = FFT.irfftn(UP/(demo + mu1/mu2), s=list(f.shape[-3:]), dim=(-3,-2,-1))
    return x

def diff_h(data):  # 水平差，列-列
    output = torch.roll(data, -1 ,dims=-1) - data
    return output

def diff_hc(data):  # 水平差，列-列
    output = torch.roll(data, 1 ,dims=-1) - data
    return output

def diff_v(data): # 垂直差， 行-行
    output = torch.roll(data, -1 ,dims=-2) - data
    return output

def diff_vc(data): # 垂直差， 行-行
    output = torch.roll(data, 1 ,dims=-2) - data
    return output

def diff_s(data): # 垂直差， 行-行
    output = torch.roll(data, -1 ,dims=-3) - data
    return output

def diff_sc(data): # 垂直差， 行-行
    output = torch.roll(data, 1 ,dims=-3) - data
    return output

def my_softmax(W):
    s1, s2, s3, s4 = W.shape
    W = nn.Softmax(-1)(W.reshape(s1,-1)).reshape(s1, s2, s3, s4)*s2*s3*s4 
    return W
    
    

            
                
 
            
