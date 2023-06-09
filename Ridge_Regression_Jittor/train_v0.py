import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import argparse
import numpy as np
import jittor as jt
import optim as optim_temp

from jittor import nn
from sklearn.model_selection import train_test_split, KFold
from model import *
from dataloader import load_data

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description='Jittor Training')
parser.add_argument('--dataset', type=str, default='abalone', help='choose which dataset to use.')
parser.add_argument('--lr', '--learning-rate', default=5e-2, type=float, help='initial learning rate')
parser.add_argument('--lr_ho', type=float, default=1e-2,
                        help='learning rate of hyperparameter')
parser.add_argument('--hyper_num', type=float, default=0.,
                        help='the initial value of hyperparameter')
parser.add_argument('--kfold', default=10, type=int, help='cross validation k fold')
parser.add_argument('--iteration', default=10000, type=int)
parser.add_argument('--innerloop', default=4, type=int, help='inner loop')
args = parser.parse_args()

# load data
train_data, train_target, test_data, test_target, feature_number = load_data(args.dataset)
train_data, train_target = jt.float32(train_data), jt.float32(train_target)
test_data, test_target = jt.float32(test_data), jt.float32(test_target)

KF = KFold(n_splits=args.kfold, shuffle=True)
x_tr, y_tr, x_val, y_val = [], [], [], []
for train_index, val_index in KF.split(train_data):
    x_tr.append(train_data[train_index])
    y_tr.append(train_target[train_index])
    x_val.append(train_data[val_index])
    y_val.append(train_target[val_index])
def build_model():
    model = Linear_Model(feature_number, 1)
    return model

mse = nn.MSELoss()

rs = jt.float32(args.hyper_num)

def reg(outputs, target, w):
    loss = mse(outputs, target) + w * rs
    return loss


net = build_model()
optimizer = jt.optim.SGD(net.parameters(), lr=args.lr)
optimizer_hyper =jt.optim.SGD([rs], lr=args.lr_ho)

model_fold = []
for _ in range(args.kfold):
    meta_model = build_model()
    model_fold.append(meta_model)

def test(model, x_te, y_te):
    with jt.no_grad():
        y_pred_te = model(x_te)
        loss_te = mse(y_pred_te, y_te)
    return loss_te

time1 = time.time()
for t in range(args.iteration):
    losstrain = 0
    lossval = 0

    for m in range(args.kfold):
        meta_model = build_model()
        optimizer_temp = optim_temp.SGD(meta_model.parameters(), args.lr)
        meta_model.load_state_dict(model_fold[m].state_dict())
        for _ in range(args.innerloop):
            pred_meta = meta_model(x_tr[m])
            for param in meta_model.parameters():
                regularization_loss = jt.norm(param)
                break
            loss = reg(pred_meta, y_tr[m], regularization_loss)
            optimizer_temp.step(loss)
        model_fold[m].load_state_dict(meta_model.state_dict())
        pred_val = meta_model(x_val[m])
        loss_val_fold = mse(pred_val, y_val[m])
        lossval += loss_val_fold
    lossval = lossval/args.kfold

    optimizer_hyper.step(lossval)

    rs.data = np.clip(rs.data, 1e-6, 1e6)
    # rs.data = jt.clamp(rs, min_v=1e-6).clone().detach()
    train_target_pred = net(train_data)
    for param in net.parameters():
        regularization_loss = jt.norm(param)
        break
    loss = mse(train_target_pred, train_target) + regularization_loss * rs.data
    optimizer.step(loss)

time2 = time.time()
loss_te = test(net, test_data, test_target)
print('train_loss: %.4f test_loss: %.4f hyper: %f cv_loss: %.4f time: %.4f'
      % (loss, loss_te, rs.data, lossval, time2-time1))