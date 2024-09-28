# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import jittor as jt
import torch
from jittor import nn
# import torch.nn.functional as F
#import matplotlib.pyplot as plt
# import sklearn.metrics as sm
# import pandas as pd
# import sklearn.metrics as sm
import random
import numpy as np
#from metrics import *
import optim as optim_temp 


from resnet32 import ResNet32
#from resnet import ResNet32,VNet
from cifar import *

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.2,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Resnet34', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.set_defaults(augment=True)

args = parser.parse_args()
jt.flags.use_cuda = 1

use_cuda = True


print()
print(args)
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden1, output)

    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return nn.Sigmoid()(out)


def accuracy(output, target):
    batch_size = target.shape[0]
    pred = np.argmax(output, -1)
    res = ( (pred == target).astype(float).sum() )/batch_size

    return res


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    return model

def accuracy(output, target):
    batch_size = target.shape[0]
    pred = np.argmax(output, -1)
    res = ( (pred == target).astype(float).sum() )/batch_size

    return res


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss +=nn.cross_entropy_loss(outputs, targets).item()
            predicted = np.argmax(outputs.detach(), -1) 
            correct += ( (predicted == targets.data).astype(float).sum() )

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader),
        accuracy))

    return accuracy


def train(train_loader,train_meta_loader,model, vnet,vnet1,optimizer_model,optimizer_vnet,optimizer_vnet1,epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0
    
    train_loss_wp = 0   
    global results
    results = np.zeros((len(train_loader.dataset), num_classes), dtype=np.float32)
    correct = 0 
    train_meta_loader.endless = True
    train_meta_loader_iter = iter(train_meta_loader)
    

    
    for batch_idx, ((inputs,inputs_u), targets,targets_true, soft_labels, indexs) in enumerate(train_loader):
        model.train()
        # print('batch_idx:', batch_idx)
        input_var = inputs # to_var(inputs, requires_grad=False)
        target_var = targets # to_var(targets, requires_grad=False).long()
        targets_true_var = targets_true # to_var(targets_true, requires_grad=False).long()
              
        if epoch < 80:
            y_f = model(input_var)
            probs = nn.softmax(y_f,dim=1)
            results[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().data.tolist()
            correct += (target_var==targets_true_var).sum().item()
            Loss = nn.cross_entropy_loss(y_f, target_var.long())
            # optimizer_model.zero_grad()
            # Loss.backward()
            optimizer_model.step(Loss)  
            prec_train = accuracy(y_f.data, target_var.long().data)
            train_loss_wp += Loss.item() 
            alpha_clean = alpha_corrupt = 0
            if (batch_idx + 1) % 100 == 0:
               print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'Prec@1 %.2f\t' % (
                      epoch , args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss_wp / (batch_idx + 1)),
                       prec_train))
            if (batch_idx + 1) % 200 == 0:
               test_acc = test(model=model, test_loader=test_loader)        
        
        else:
            

            index_all = np.arange(args.batch_size)
            index_clean = np.where(np.array(targets.cpu()) == np.array(targets_true.cpu()))

            index_clean = np.array(index_clean)
            index_clean = [i for y in index_clean for i in y]
            
            index_corrupt = np.where(np.array(targets.cpu()) != np.array(targets_true.cpu()))
            index_corrupt = np.array(index_corrupt)
            index_corrupt = [i for y in index_corrupt for i in y]

            meta_model = build_model()        
            meta_model.load_state_dict(model.state_dict())
            meta_lr = args.lr * ((0.1 ** int(epoch >= 100))) 
            optimizer_temp = optim_temp.SGD(meta_model.parameters(), meta_lr,
                                            momentum=args.momentum, weight_decay=args.weight_decay)

            y_f_hat = meta_model(input_var)
            
            z = torch.max(soft_labels,dim=1)[1].long()

            cost = nn.cross_entropy_loss(y_f_hat, target_var, reduction='none')
            cost_v = torch.reshape(cost, (len(cost), 1))
        
            l_lambda = vnet(cost_v.detach())
            
            cost1 = nn.cross_entropy_loss(y_f_hat, target_var, reduction='none')
            cost_v1 = torch.reshape(cost1, (len(cost1), 1))
            l1 = torch.sum(cost_v1*l_lambda)/len(cost_v1)
            
            cost2 = nn.cross_entropy_loss(y_f_hat, z, reduction='none')
            cost_v2 = torch.reshape(cost2, (len(cost2), 1))   
            lambda1 = vnet1(cost_v2.detach())
            
            
            current_label = torch.max(y_f_hat,dim=1)[1]
            cost3 = nn.cross_entropy_loss(y_f_hat,current_label, reduction='none')
            cost_v3 = torch.reshape(cost3,(len(cost3),1))
            l2 = torch.sum(cost_v2*(lambda1)*(1-l_lambda))/len(cost_v2)+torch.sum(cost_v3*(1-lambda1)*(1-l_lambda))/len(cost_v3)
            l_f_meta = l1 +l2
            
            optimizer_vnet.step(l_f_meta)

            # meta_model.zero_grad()
            # grads = torch.autograd.grad(l_f_meta,(meta_model.params()),create_graph=True)
            # meta_lr = args.lr * ((0.1 ** int(epoch >= 100))) 
            # meta_model.update_params(lr_inner=meta_lr,source_params=grads)
            # del grads
            inputs_val, targets_val = next(train_meta_loader_iter)
            y_g_hat = meta_model(inputs_val)
            l_g_meta = nn.cross_entropy_loss(y_g_hat, targets_val)

            optimizer_vnet.step(l_g_meta)
            optimizer_vnet1.step(l_g_meta)
            prec_meta = 0.
        
            # y_g_hat = meta_model(input_validation_var)
            # l_g_meta = nn.cross_entropy_loss(y_g_hat, target_validation_var)
            # prec_meta = accuracy(y_g_hat.data, target_validation_var.data)
            
            # optimizer_vnet.zero_grad()
            # optimizer_vnet1.zero_grad()
            # l_g_meta.backward()
            # optimizer_vnet.step()
            # optimizer_vnet1.step()
            
            y_f1 = model(input_var)  
            probs = nn.softmax(y_f1,dim=1)

            
            cost_w = nn.cross_entropy_loss(y_f1, target_var, reduction='none')
            cost_v21 = torch.reshape(cost_w, (len(cost_w), 1))
            prec_train = accuracy(y_f1.data, target_var.data)
            
            cost_w1 = nn.cross_entropy_loss(y_f1, z, reduction='none')
            cost_v22 = torch.reshape(cost_w1, (len(cost_w1), 1))
            
            cost_w2 = nn.cross_entropy_loss(y_f1,torch.max(y_f1,dim=1)[1], reduction='none')
            cost_v23 = torch.reshape(cost_w2, (len(cost_w2), 1)) 
            #print(cost_v23)
            with torch.no_grad():
                w_v = vnet(cost_v21) 
                w_v2 = vnet1(cost_v22)

                
            loss1 = torch.sum(w_v*cost_v21)/len(cost_v21)     
            
            loss2 = torch.sum(cost_v22*w_v2*(1-w_v))/len(cost_v22)+torch.sum(cost_v23*(1-w_v2)*(1-w_v))/len(cost_v23) 
            
            
            new_pseudolabel = (w_v2*soft_labels.float())+((1-w_v2)*probs)                    
            #new_pseudolabel_oh = torch.zeros(inputs.size()[0], num_classes).scatter_(1, new_pseudolabel_hard.cpu().view(-1,1), 1)
            #results[indexs.cpu().detach().numpy().tolist()] = new_pseudolabel.cpu().detach().numpy().tolist()
            # target_var_oh = torch.zeros(inputs.size()[0], num_classes).scatter_(1, targets.view(-1,1), 1)
            new_label = new_pseudolabel*(1-w_v) #+ w_v*target_var_oh
            results[indexs.cpu().detach().numpy().tolist()] = new_label.cpu().detach().numpy().tolist()            
            correct_label = torch.max(new_label,1)[1]
            correct += (targets_true_var==correct_label).sum().item()            

            Loss = loss1+loss2 
            
            # optimizer_model.zero_grad()
            # Loss.backward()
            optimizer_model.step(Loss)           
  
        
            train_loss += Loss.item() 
            meta_loss += l_g_meta.item()
            if (batch_idx + 1) % 100 == 0:
               print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))
            if (batch_idx + 1) % 200 == 0:
               test_acc = test(model=model, test_loader=test_loader)
    train_loader.dataset.label_update(results)    
    

# train_loader, train_meta_loader, test_loader = build_dataset()
train_loader, train_meta_loader, test_loader = build_dataset(args)

# create model
model = build_model()

vnet = VNet(1, 100, 1)
vnet1 = VNet(1, 100, 1)


if args.dataset == 'cifar10':
    num_classes = 10
if args.dataset == 'cifar100':
    num_classes = 100


optimizer_model = jt.optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_vnet = jt.optim.Adam(vnet.parameters(), 1e-3,
                             weight_decay=1e-4)
optimizer_vnet1 = jt.optim.Adam(vnet1.parameters(), 1e-3,
                             weight_decay=1e-4)

def evaluate(results, train_loader, evaluator):
    model.eval()
    correct = 0
    test_loss = 0
    evaluator.reset()
    with torch.no_grad():
        #for batch_idx, (inputs, targets) in enumerate(test_loader):
        for batch_idx, ((inputs,inputs_u), targets,targets_true,soft_labels,indexs) in enumerate(train_loader):
            outputs = model(inputs)
            #pred = torch.max(outputs,dim=1)[1]#
            evaluator.add_batch(targets_true,results)
    return evaluator.confusion_matrix 

def main():
    best_acc = 0
 
    for epoch in range(1,args.epochs+1):
        adjust_learning_rate(optimizer_model, epoch)     

        train(train_loader,train_meta_loader,model, vnet,vnet1,optimizer_model,optimizer_vnet,optimizer_vnet1,epoch)
        test_acc = test(model=model, test_loader=test_loader) 
        if test_acc >= best_acc:
            best_acc = test_acc
    print('best accuracy:', best_acc)


if __name__ == '__main__':
    main()
