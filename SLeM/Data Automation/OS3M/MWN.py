import argparse
import numpy as np
import jittor as jt
import optim as optim_temp 

from data import *
from jittor import nn
from model import *
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from torch.distributions import MultivariateNormal


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--model', default='resnet32', type=str, help='model')
parser.add_argument('--num_classes', default=10, type=int, help='the number of dataset classes')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int, help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--prefetch', type=int, default=1, help='Pre-fetching threads.')
parser.add_argument('--corruption_prob', type=float, default=0.4, help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif', help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--on', type=float, default=0.4, help='ood noise')
parser.add_argument('--noisy_dataset', type=str, default='cifar10')
parser.add_argument('--noise_data_dir', type=str, default='./')


args = parser.parse_args()

jt.flags.use_cuda = 1

model_dict = {'resnet32':ResNet32}


def build_model():
    model = model_dict[args.model](num_classes=args.num_classes)
    return model


class VNet(nn.Module):
    def __init__(self, inputs, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(inputs, hidden1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden1, output)

    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return nn.Sigmoid()(out)


class ACVNet_N(nn.Module):
    def __init__(self):
        super(ACVNet_N, self).__init__()
        self.layers0 = VNet(1, 100, 1)

    def forward(self, x_all):
        output = self.layers0( x_all )
        
        return output
    

class ACVNet_O(nn.Module):
    def __init__(self):
        super(ACVNet_O, self).__init__()
        self.layers0 = VNet(1, 100, 1)

    def forward(self, x_all):
        output = self.layers0( x_all )
 
        return output
    

def accuracy(output, target):
    batch_size = target.shape[0]
    pred = np.argmax(output, -1)
    res = ( (pred == target).astype(float).sum() )/batch_size

    return res


def print_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print('\n Epoch:{:.4f}|LR:{:.4f}\n'.format(epoch,lr))
    return lr
  

def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ce_loss(output, target, reduce=True):
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))
    target = target.reshape((-1, ))
    target = target.broadcast(output, [1])
    target = target.index(1) == target
    
    output = output - output.max([1], keepdims=True)
    loss = output.exp().sum(1).log()
    loss = loss - (output*target).sum(1)
    if reduce:
        return loss.mean()
    else:
        return loss


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    for _, (inputs, targets, _) in enumerate(test_loader):
        inputs, targets = jt.array(inputs), jt.array(targets)
        outputs, _ = model(inputs)
        test_loss += nn.cross_entropy_loss(outputs, targets).detach().item()
        predicted = np.argmax(outputs.detach(), -1) 
        correct += ( (predicted == targets.data).astype(float).sum() )

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader),
        accuracy))

    return accuracy


def train(train_loader, train_meta_loader, model, vnet1, vnet2, optimizer_model, optimizer_vnet1, optimizer_vnet2, epoch, meta_lr):
    print('\nEpoch: %d' % epoch)

    model.eval()
    with jt.no_grad():        
        all_out_s = jt.array([]) 
        true_label_s = jt.array([]) 
        for inputs_s, targets_s, _ in train_meta_loader:
            _, feat_s  = model(inputs_s)

            all_out_s = jt.cat((all_out_s, feat_s), dim=0)
            true_label_s = jt.cat((true_label_s, targets_s), dim=0)

        prototypes = jt.array([]) 
        for i in range(args.num_classes):
            prototypes = jt.cat( (prototypes, all_out_s[true_label_s==i].mean(0).unsqueeze(0) ), dim=0)


    train_loss = 0
    meta_loss = 0

    train_meta_loader.endless = True
    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets, index) in enumerate(train_loader):
        model.train()

        meta_model = build_model()
        meta_model.load_state_dict(model.state_dict())
        optimizer_temp = optim_temp.SGD(meta_model.parameters(), meta_lr,
                                        momentum=args.momentum, weight_decay=args.weight_decay)

        outputs, _ = meta_model(inputs)
        cost = ce_loss(outputs, targets, reduce=False).reshape((-1,1))
        
        with jt.no_grad():
            # all_logit = jt.nn.CosineSimilarity(dim=2, eps=1e-6)( f_t.unsqueeze(1).repeat(1, args.num_class, 1), prototypes.repeat(f_t.shape[0], 1, 1) )
            
            vec1 = f_t.unsqueeze(1).repeat(1, args.num_classes, 1)
            vec2 = prototypes.repeat(f_t.shape[0], 1, 1)

            dot_product = jt.sum(vec1 * vec2, dim=-1)
            norm_vec1 = jt.norm(vec1, dim=-1)
            norm_vec2 = jt.norm(vec2, dim=-1)
            all_logit = dot_product / (norm_vec1 * norm_vec2)

            cost_2 = 5. - jt.max(all_logit, dim=-1) * 5.
            cost_1 = ce_loss(all_logit * 2, targets, reduce=False).reshape((-1,1)) 

        v_lambda1 = vnet1(cost_1.detach())
        v_lambda2 = vnet2(cost_2.detach())

        l_f_meta = (cost * v_lambda1 * v_lambda2 ).mean() 
        optimizer_temp.step(l_f_meta)

        inputs_val, targets_val, _ = next(train_meta_loader_iter)
        y_g_hat, _ = meta_model(inputs_val)
        l_g_meta = nn.cross_entropy_loss(y_g_hat, targets_val) + (( cost_2.detach() * v_lambda2 ).sum() - (sel_t * v_lambda2).sum()) * 100.

        optimizer_vnet1.step(l_g_meta, retain_graph=True)
        optimizer_vnet2.step(l_g_meta)

        outputs, _ = model(inputs)
        cost_w = ce_loss(outputs, targets, reduce=False).reshape((-1, 1))
        
        with jt.no_grad():
            # all_logit = jt.nn.CosineSimilarity(dim=2, eps=1e-6)( f_t.unsqueeze(1).repeat(1, args.num_class, 1), prototypes.repeat(f_t.shape[0], 1, 1) )

            vec1 = f_t.unsqueeze(1).repeat(1, args.num_classes, 1)
            vec2 = prototypes.repeat(f_t.shape[0], 1, 1)

            dot_product = jt.sum(vec1 * vec2, dim=-1)
            norm_vec1 = jt.norm(vec1, dim=-1)
            norm_vec2 = jt.norm(vec2, dim=-1)
            all_logit = dot_product / (norm_vec1 * norm_vec2)

            cost_2 = 5. - jt.max(all_logit, dim=-1) * 5.
            cost_1 = ce_loss(all_logit * 2, targets, reduce=False).reshape((-1,1)) 
            
            v_lambda1 = vnet1(cost_1)
            v_lambda2 = vnet2(cost_2)

        loss = (cost_w * v_lambda1 * v_lambda2).mean()

        optimizer_model.step(loss)

        with jt.no_grad(no_fuse=1):
            train_loss += loss
            meta_loss += l_g_meta

        if (batch_idx + 1) % 50 == 0:
            prec_meta = accuracy(y_g_hat.data, targets_val.data)
            prec_train = accuracy(outputs.data, targets.data)
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader), (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))


# load dataset 
train_loader, train_meta_loader, test_loader = build_dataset(args)

# load model
model = build_model()
vnet1 = VNet(1, 100, 1)
vnet2 = VNet(1, 100, 1)

optimizer_model = jt.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_vnet1 = jt.optim.Adam(vnet1.parameters(), 1e-3, weight_decay=1e-4)
optimizer_vnet2 = jt.optim.Adam(vnet2.parameters(), 1e-3, weight_decay=1e-4)


def main():
    best_acc = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch)
        meta_lr = print_lr(optimizer_model, epoch)
        train(train_loader,train_meta_loader,model, vnet1, vnet2, optimizer_model, optimizer_vnet1, optimizer_vnet2, epoch, meta_lr)
        test_acc = test(model=model, test_loader=test_loader)
        # vnet.save('./save/vnet_%d.pkl' % epoch)
        if test_acc >= best_acc:
            best_acc = test_acc

    print('best accuracy:', best_acc)


if __name__ == '__main__':
    main()
