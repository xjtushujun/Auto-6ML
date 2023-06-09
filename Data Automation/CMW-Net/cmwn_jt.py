import os
import sys
import argparse

import jittor as jt
from jittor import nn
import numpy as np
from sklearn.cluster import KMeans
import optim as optim_temp 

from PreResNet_jt import *
import dataloader_cifar_jt as dataloader


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--ema', default=0.997, type=float)
parser.add_argument('--noise_mode',  default='fd')
parser.add_argument('--alpha', default=1., type=float, help='parameter for Beta')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default='./cifar100-python', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar100', type=str)
args = parser.parse_args()

jt.flags.use_cuda = 1


# Training
def accuracy(output, target):
    batch_size = target.shape[0]
    pred = np.argmax(output, -1)
    res = ( (pred == target).astype(float).sum() )/batch_size

    return res


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** int(epoch >= 150))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print('\n Epoch:{:.4f}|LR:{:.4f}\n'.format(epoch,lr))
    return lr


def ce_loss(output, target, reduce=True):
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))

    if len(target.shape) == 1: 
        target = target.reshape((-1, ))
        target = target.broadcast(output, [1])
        target = target.index(1) == target
    
    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = logsum - (output*target).sum(1)
    if reduce:
        return loss.mean()
    else:
        return loss


def warmup(epoch,model,model_ema,vnet,optimizer_model,optimizer_vnet,train_loader, train_meta_loader, meta_lr):
    num_iter = (len(train_loader)//args.batch_size)+1

    train_meta_loader.endless = True
    train_meta_loader_iter = iter(train_meta_loader)

    for batch_idx, (inputs, targets, index) in enumerate(train_loader):
        model.train()
        model_ema.train()

        with jt.no_grad():
            outputs_ema = model_ema(inputs)
            psudo_label = nn.softmax(outputs_ema, dim=1)

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        idx = np.random.permutation(inputs.shape[0])
        mix_inputs = l * inputs + (1-l) * inputs[idx]

        if batch_idx % 10 == 0:
            meta_model = create_model()
            meta_model.load_state_dict(model.state_dict())

            optimizer_temp = optim_temp.SGD(meta_model.parameters(), meta_lr, momentum=0.9, weight_decay=5e-4)

            outputs = meta_model(mix_inputs)

            cost_1 = ce_loss(outputs, targets, reduce=False).reshape((-1,1))
            v_lambda_1 = vnet(cost_1.detach(), targets.detach(), c).reshape((-1,))

            cost_2 = ce_loss(outputs[idx], targets[idx], reduce=False).reshape((-1,1))
            v_lambda_2 = vnet(cost_2.detach(), targets[idx].detach(), c).reshape((-1,))


            l_f_meta = ( l * ( ce_loss(outputs, targets, reduce=False) * v_lambda_1 
                             + ce_loss(outputs, psudo_label, reduce=False) * (1-v_lambda_1)
                              )
                   + (1-l) * ( ce_loss(outputs, targets[idx], reduce=False) * v_lambda_2
                             + ce_loss(outputs, psudo_label[idx], reduce=False) * (1 - v_lambda_2)
                              )
                        ).mean()

            optimizer_temp.step(l_f_meta)

            inputs_val, targets_val = next(train_meta_loader_iter)

            ll = np.random.beta(1., 1.)
            ll = max(ll, 1-ll)
            idxx = np.random.permutation(inputs_val.shape[0])
            mix_inputs_val = ll * inputs_val + (1-ll) * inputs_val[idxx]

            y_g_hat = meta_model(mix_inputs_val)
             
            l_g_meta = ll * ce_loss(y_g_hat, targets_val) + (1-ll) * ce_loss(y_g_hat, targets_val[idxx])

            optimizer_vnet.step(l_g_meta)

        outputs = model(mix_inputs)

        with jt.no_grad():
            cost_1 = ce_loss(outputs, targets, reduce=False).reshape((-1,1))
            cost_2 = ce_loss(outputs[idx], targets[idx], reduce=False).reshape((-1,1))

            v_lambda_1 = vnet(cost_1, targets, c).reshape((-1,))
            v_lambda_2 = vnet(cost_2, targets[idx], c).reshape((-1,))

        loss = ( l * ( ce_loss(outputs, targets, reduce=False) * v_lambda_1 
                     + ce_loss(outputs, psudo_label, reduce=False) * (1-v_lambda_1)
                      )
           + (1-l) * ( ce_loss(outputs, targets[idx], reduce=False) * v_lambda_2
                     + ce_loss(outputs, psudo_label[idx], reduce=False) * (1 - v_lambda_2)
                      )
                ).mean()
    
        optimizer_model.step(loss)

        with jt.no_grad():
            for (param, param_ema) in zip(model.parameters(), model_ema.parameters()):
                # print('param:', param)
                param_ema.assign( param_ema.multiply(args.ema) + param.multiply(1-args.ema) )
                # param_ema = param_ema.multiply(args.ema) + param.multiply(1-args.ema)

        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t' % (
                      (epoch + 1), args.num_epochs, batch_idx + 1, len(train_loader) ) )


def train_CE(train_loader, model, model_ema, optimizer,epoch):
    print('\nEpoch: %d' % epoch)
    
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        model.train()
        model_ema.train()

        outputs = model(inputs)
        loss = ce_loss(outputs, targets)

        optimizer.step(loss)

        with jt.no_grad():
            for (param, param_ema) in zip(model.parameters(), model_ema.parameters()):
                param_ema.assign( param_ema.multiply(args.ema) + param.multiply(1-args.ema) )
                # param_ema.mul_(args.ema).add_(1-args.ema, param)

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(inputs), len(train_loader), 100. * batch_idx / len(train_loader) ))


        
def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with jt.no_grad():
      for _, (inputs, targets) in enumerate(test_loader):
        inputs, targets = jt.array(inputs), jt.array(targets)
        outputs = model(inputs)
        test_loss += nn.cross_entropy_loss(outputs, targets)
        predicted = np.argmax(outputs.detach(), -1) 
        correct += ( (predicted == targets.data).astype(float).sum() )

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader),
        accuracy))

    return accuracy
    

def eval_train(model):
    print('estimating loss ...')
    model.eval()
    num_iter = (len(eval_loader) // args.batch_size) + 1
    losses = jt.zeros( (len(eval_loader), ) )
    paths = []
    n = 0
    with jt.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            outputs = model(inputs)
            loss = ce_loss(outputs, targets, reduce=False)
            for b in range(inputs.shape[0]):
                losses[n] = loss[b]
                paths.append(path[b])
                n += 1

    return losses, paths


def create_model():
    model = ResNet18(num_classes=args.num_class)
    return model


if args.dataset=='cifar10':
    warm_up = 10
    args.num_class = 10
elif args.dataset=='cifar100':
    warm_up = 30
    args.num_class = 100

vnet = ACVNet(1, 100, 1, 3)
print('end')
loader = dataloader.cifar_dataloader(data_name=args.dataset, batch_size=args.batch_size, num_workers=4, root_dir=args.data_path,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net = create_model()
net_ema = create_model()

optimizer = jt.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

test_loader = loader.run('test')
warmup_trainloader = loader.run('warmup')
eval_loader = loader.run('eval_train')

with jt.no_grad():
    a = []
    web_num = warmup_trainloader.noise_label
    for i in range(args.num_class):
        a.append([web_num.count(i)])

    print(len(web_num))

    print(a)
    es = KMeans(3)
    es.fit(a)

    c = es.labels_

    print('c:', c.tolist())

    w = [[],[],[]]
    for i in range(3):
        for k, j in enumerate(c):
            if i == j:
                w[i].append(a[k][0])

    print(w)

    net_ema.load_state_dict(net.state_dict())


vnet = ACVNet(1, 100, 1, 3)
optimizer_vnet = jt.optim.Adam(vnet.parameters(), 1e-4, weight_decay=1e-4)

test_acc = test(net, test_loader)  
test_acc_ema = test(net_ema, test_loader)  


for epoch in range(args.num_epochs):
    adjust_learning_rate(optimizer, epoch)
    if epoch < warm_up:       
        print('Warmup Net')
        train_CE(warmup_trainloader, net, net_ema, optimizer, epoch)

    else: 
        if args.noise_mode == 'asym' and epoch > 149:
            args.ema = 1.

        losses, _ = eval_train(net_ema)
        print('losses:', type(losses), losses.shape)

        train_imagenet_loader = loader.run('meta', losses)

        meta_lr = print_lr(optimizer, epoch)
 
        warmup(epoch,net,net_ema,vnet,optimizer,optimizer_vnet,warmup_trainloader,train_imagenet_loader, meta_lr)


    test_acc = test(net, test_loader)  
    test_acc_ema = test(net_ema, test_loader)  

    print("\n| Test Epoch #%d\t Test Acc: %.2f%% \n"%(epoch,test_acc))  
    print("\n| Test Epoch #%d\t Test ema Acc: %.2f%% \n"%(epoch,test_acc_ema))  



