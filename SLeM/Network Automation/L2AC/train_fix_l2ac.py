import argparse
import os
import shutil
import time
import random
import numpy as np
import jittor as jt
import torch
from jittor import nn
# import torch.nn.parallel
# import torch.utils.data as data
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
import optim as optim_temp 

import models.wideresnet_l2ac as models
# from models.meta_adam import MetaAdam
from dataset import fix_cifar10, fix_cifar100, fix_stl10
from dataset.ClassAwareSampler import ClassAwareSampler
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
# Dataset options
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
# Miscs
parser.add_argument('--seed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--workers', type=int, default=4, metavar='N',
                    help='dataloader threads')
# Method options
parser.add_argument('--num_max', type=int, default=1500,
                    help='number of samples in the maximal class')
parser.add_argument('--ratio', type=float, default=2.0,
                    help='relative size between labeled and unlabeled data')
parser.add_argument('--imb_ratio_l', type=int, default=100,
                    help='imbalance ratio for labeled data')
parser.add_argument('--imb_ratio_u', type=int, default=None,
                    help='imbalance ratio for unlabeled data')
parser.add_argument('--step', action='store_true',
                    help='type of class-imbalance')
parser.add_argument('--reverse', type=bool, default=False,
                    help="choose for reverse the distribution of unlabeled data")
parser.add_argument('--val-iteration', type=int, default=500,
                    help='Frequency for the evaluation')
parser.add_argument('--num_val', type=int, default=10,
                    help='Number of validation data')
# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.95, type=float,
                    help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)
args = parser.parse_args()

jt.flags.use_cuda = 1

use_cuda = True

state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# use_cuda = torch.cuda.is_available()

# Random seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
np.random.seed(args.seed)

best_acc = 0  # best test accuracy
if args.dataset == 'cifar100':
    num_class = 100
else:
    num_class = 10


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio_l)
    if args.imb_ratio_u is not None:
        U_SAMPLES_PER_CLASS = make_imb_data(args.ratio * args.num_max, num_class, args.imb_ratio_u)
    if args.reverse:
        args.imb_ratio_u = 1.0 / args.imb_ratio_u
        U_SAMPLES_PER_CLASS.reverse()
        print(f'==> Reverse the distribution of unlabeled data')

    if args.dataset == 'cifar100':
        print(f'==> Preparing imbalanced CIFAR-100')
        train_labeled_set, train_unlabeled_set, test_set = fix_cifar100.get_cifar100(
            './data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)
    elif args.dataset == 'stl10':
        print(f'==> Preparing imbalanced STL-10')
        train_labeled_set, train_unlabeled_set, test_set = fix_stl10.get_stl10(
            './data', N_SAMPLES_PER_CLASS)
    else:
        print(f'==> Preparing imbalanced CIFAR-10')
        train_labeled_set, train_unlabeled_set, test_set, train_labeled_dataset1 = fix_cifar10.get_cifar10(
            './data/', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)

    labeled_trainloader = train_labeled_set.set_attrs(batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    unlabeled_trainloader = train_unlabeled_set.set_attrs(batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = test_set.set_attrs(batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print('labeled_trainloader:', len(labeled_trainloader), len(unlabeled_trainloader), len(test_loader) )

    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.MetaWideResNet(num_classes=num_class)
        # model = model.cuda()

        if ema:
            for param in model.parameters():
                # param.detach_()
                param = param.detach()

        return model

    model = create_model()
    ema_model = create_model(ema=True)
    # cudnn.benchmark = True
    print('==> Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    train_criterion = SemiLoss()
    # criterion = nn.CrossEntropyLoss()

    classifier_params = list(map(id, model.fc.parameters()))
    feat_params = list(filter(lambda p: id(p) not in classifier_params,
                     model.parameters()))

    optimizer_feat = jt.optim.Adam(feat_params, lr=args.lr)
    optimizer_fc = jt.optim.Adam(model.fc.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)

    meta_net = models.MetaNet(in_dim=num_class, out_dim=num_class, hid_dim=256).cuda()
    optimizer_meta = jt.optim.Adam(meta_net.parameters(), 1e-4)
    print('==>Total params: %.2fM' % (sum(p.numel() for p in meta_net.parameters()) / 1000000.0))

    # Resume
    start_epoch = 0
    title = 'fix-cifar-10'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Test Loss', 'Test Acc.', 'Test GM.'])

    test_accs = []
    test_gms = []


    # for (inputs_x, targets_x, _) in labeled_trainloader:
    #     print('inputs_x, targets_x, labeled_trainloader:', inputs_x, targets_x)


    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        if epoch == start_epoch or epoch % 10 == 0:
            # meta_loader = train_labeled_dataset1.set_attrs(batch_size=args.batch_size, sampler=ClassAwareSampler(train_labeled_set, 10), shuffle=False, num_workers=args.workers, drop_last=True)
            meta_loader = train_labeled_dataset1.set_attrs(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

        train_loss, train_loss_x, train_loss_u = train(
            labeled_trainloader, unlabeled_trainloader, meta_loader, model, meta_net, optimizer_feat,
            optimizer_fc, ema_optimizer, optimizer_meta, train_criterion, epoch, use_cuda)

        # Evaluation part
        test_loss, test_acc, test_cls, test_gm = validate(
            test_loader, ema_model, mode='Test Stats ')

        # Append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, test_loss, test_acc, test_gm])

        # Save models
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
        }, epoch + 1)
        test_accs.append(test_acc)
        test_gms.append(test_gm)

    logger.close()

    # Print the final results
    print('Mean bAcc:')
    print(np.mean(test_accs[-20:]))

    print('Mean GM:')
    print(np.mean(test_gms[-20:]))

    print('Name of saved folder:')
    print(args.out)


def train(labeled_trainloader, unlabeled_trainloader, meta_loader, model, meta_net, optimizer_feat, optimizer_fc,
          ema_optimizer, optimizer_m, criterion, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    loss_meta = 0
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    meta_iter = iter(meta_loader)

    model.train()
    for batch_idx in range(args.val_iteration):
        # print('labeled_train_iter: batch_idx:', batch_idx, len(labeled_trainloader))

        # print('labeled_train_iter: batch_idx:', batch_idx, labeled_trainloader, next(labeled_train_iter))
        try:
            inputs_x, targets_x, _ = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)

        try:
            inputs_m, targets_m, _ = next(meta_iter)
        except:
            meta_iter = iter(meta_loader)
            inputs_m, targets_m, _ = next(meta_iter)

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        # targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1, 1).long(), 1)
        targets_x = nn.one_hot(targets_x, num_class)
        # if use_cuda:
        #     inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        #     inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()
        #     inputs_m, targets_m = inputs_m.cuda(), targets_m.cuda()

        # build meta_model
        meta_model = models.MetaWideResNet(num_classes=num_class)
        meta_model.load_state_dict(model.state_dict())
        # meta_optimizer = MetaAdam(meta_model.fc, lr=args.lr)
        # meta_optimizer.load_state_dict(optimizer_fc.state_dict())
        meta_optimizer = optim_temp.Adam(meta_model.parameters(), args.lr)


        # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            outputs_u = meta_model(inputs_u)
            if args.imb_ratio_l == args.imb_ratio_u:
                pseudo_labels = jt.normalize(outputs_u, dim=-1).detach()
                res_outputs = meta_net(pseudo_labels, pseudo_labels)
                outputs_u = outputs_u + res_outputs[batch_size:]
            targets_u = nn.softmax(outputs_u, dim=1)

        max_p, p_hat = torch.max(targets_u, dim=1)
        # p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        p_hat = nn.one_hot(p_hat, num_class)

        # select_mask = max_p.ge(max_p)
        select_mask = max_p > max_p
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, p_hat, p_hat], dim=0)

        if batch_idx % 10 == 0:
            all_outputs = meta_model(all_inputs)
            pseudo_labels = jt.normalize(all_outputs, dim=-1).detach()
            res_outputs = meta_net(pseudo_labels[:batch_size], pseudo_labels[batch_size:])
            all_outputs = all_outputs + res_outputs
            logits_x = all_outputs[:batch_size]
            logits_u = all_outputs[batch_size:]
            Lx_hat, Lu_hat = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:],
                                       select_mask)
            loss_hat = Lx_hat + Lu_hat

            # meta_model.zero_grad()
            # grads = torch.autograd.grad(loss_hat, (meta_model.fc.params()), create_graph=True, allow_unused=True)
            meta_optimizer.step(loss_hat)
            # del grads

            logits_meta_hat = meta_model(inputs_m)
            loss_meta = nn.cross_entropy_loss(logits_meta_hat, targets_m.long())

            # optimizer_m.zero_grad()
            # loss_meta.backward()
            optimizer_m.step(loss_meta)

        all_outputs = model(all_inputs)
        with torch.no_grad():
            pseudo_labels = jt.normalize(all_outputs, dim=-1).detach()
            res_outputs = meta_net(pseudo_labels[:batch_size], pseudo_labels[batch_size:])
        all_outputs = all_outputs + res_outputs
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]
        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        # optimizer_feat.zero_grad()
        # optimizer_fc.zero_grad()
        # loss.backward()
        optimizer_feat.step(loss)
        optimizer_fc.step(loss)
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                     'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Loss_m: {loss_meta:.4f}' \
            .format(batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_meta=loss_meta)
        bar.next()
    bar.finish()

    return losses.avg, losses_x.avg, losses_u.avg


def validate(valloader, model, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_class).cuda()
    classwise_num = torch.zeros(num_class).cuda()
    section_acc = np.zeros(3)

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # if use_cuda:
            #     inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = nn.cross_entropy_loss(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs.max(1)[1]
            pred_mask = (targets == pred_label).float()
            for i in range(num_class):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}' \
                .format(batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg)
            bar.next()
        bar.finish()

    # Major, Neutral, Minor
    section_num = int(num_class / 3)
    classwise_acc = (classwise_correct / classwise_num).cpu().numpy()
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()
    GM = 1
    for i in range(num_class):
        if classwise_acc[i] == 0:
            # To prevent the N/A values, we set the minimum value as 0.001
            GM *= (1 / (100 * num_class)) ** (1 / num_class)
        else:
            GM *= (classwise_acc[i]) ** (1 / num_class)

    return losses.avg, top1.avg, section_acc, GM


def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1 / gamma, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    print(class_num_list)
    return list(class_num_list)


def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(nn.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(nn.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)
        # Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1))

        return Lx, Lu


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.assign(ema_param)
            # param_ema.assign( param_ema.multiply(args.ema) + param.multiply(1-args.ema) )

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            # ema_param.mul_(self.alpha)
            # ema_param.add_(param * one_minus_alpha)
            # # customized weight decay
            # param.mul_(1 - self.wd)
            ema_param.assign( ema_param.multiply(self.alpha) + param.multiply(one_minus_alpha) )
            ema_param.assign( ema_param.multiply(1 - self.wd) )


if __name__ == '__main__':
    main()


