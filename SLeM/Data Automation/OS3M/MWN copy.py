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
        x1 = (x_all[:, 0].unsqueeze(1))  
        output = self.layers0( x1 )
        
        return output
    

class ACVNet_O(nn.Module):
    def __init__(self):
        super(ACVNet_O, self).__init__()
        self.layers0 = VNet(1, 100, 1)

    def forward(self, x_all):
        x1 = x_all[:, 1].unsqueeze(1)
        output = self.layers0( x1 )
 
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

    for _, (inputs, targets) in enumerate(test_loader):
        inputs, targets = jt.array(inputs), jt.array(targets)
        outputs = model(inputs)
        test_loss += nn.cross_entropy_loss(outputs, targets).detach().item()
        predicted = np.argmax(outputs.detach(), -1) 
        correct += ( (predicted == targets.data).astype(float).sum() )

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader),
        accuracy))

    return accuracy


def train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch, meta_lr):
    print('\nEpoch: %d' % epoch)






    with jt.no_grad():
        all_con = jt.zeros(len(train_loader.dataset)).float()
        all_loss = jt.zeros(len(train_loader.dataset)).float()

        for inputs_u, targets, index in train_loader:
            _, f_u  = model(inputs_u)
            all_logit = jt.nn.CosineSimilarity(dim=2, eps=1e-6)( f_u.unsqueeze(1).repeat(1, args.num_class, 1), all_centers.repeat(f_u.shape[0], 1, 1) )

            all_con[index] = 5. - jt.max(all_logit, dim=-1) * 5
            all_loss[index] = nn.cross_entropy_loss(all_logit * 2, targets.long(), reduction='none')


        # ###############################################################################################################################

        temp_c = KMeans(2)
        temp_c.fit(all_con.reshape(-1, 1).numpy())
        thro = temp_c.cluster_centers_.reshape(-1)
        if thro[0]>thro[1]:
            sel_t = ( min(all_con[temp_c.labels_==0]).item() + max(all_con[temp_c.labels_==1]).item() )/2.
        else:
            sel_t = ( min(all_con[temp_c.labels_==1]).item() + max(all_con[temp_c.labels_==0]).item() )/2.

        # print('sel_t:', sel_t )
        # sel_t_a.append(sel_t)




        sel_t_in = ( jt.array(sel_t).repeat(len(all_con), 1) ).sum(1)
        sel_t_in = sel_t_in.cuda()

        sel_t_all.append(sel_t_in)


        # #######################################################################################################################
        loss_con = jt.cat((all_loss.reshape(-1, 1), all_con.reshape(-1, 1) ), dim=1)

        gmm_loss_con = GaussianMixture(n_components=6,max_iter=10,tol=1e-2,reg_covar=5e-4)
        gmm_loss_con.fit( loss_con.cpu() )


        mvn0 = MultivariateNormal( torch.tensor( ((gmm_loss_con.means_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :])[0] ), \
                                torch.tensor( ( (gmm_loss_con.covariances_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :, :] )[0] ) )
        mvn1 = MultivariateNormal( torch.tensor( ((gmm_loss_con.means_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :])[1] ), \
                                torch.tensor( ((gmm_loss_con.covariances_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :, :])[1] ) )
        mvn2 = MultivariateNormal( torch.tensor( ((gmm_loss_con.means_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :])[2] ), \
                                torch.tensor( ((gmm_loss_con.covariances_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :, :])[2] ) )
        mvn3 = MultivariateNormal( torch.tensor( ((gmm_loss_con.means_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :])[3] ), \
                                torch.tensor( ( (gmm_loss_con.covariances_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :, :] )[3] ) )
        mvn4 = MultivariateNormal( torch.tensor( ((gmm_loss_con.means_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :])[4] ), \
                                torch.tensor( ((gmm_loss_con.covariances_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :, :])[4] ) )
        mvn5 = MultivariateNormal( torch.tensor( ((gmm_loss_con.means_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :])[5] ), \
                                torch.tensor( ((gmm_loss_con.covariances_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist(), :, :])[5] ) )

        lambda_all = ((gmm_loss_con.weights_)[(gmm_loss_con.means_)[:, 0].reshape(-1).argsort().tolist()])

        prob_dima = (mvn0.log_prob( loss_con.cpu() )).exp() 
        prob_dimb = (mvn1.log_prob( loss_con.cpu() )).exp() 
        prob_dimc = (mvn2.log_prob( loss_con.cpu() )).exp() 
        prob_dimd = (mvn3.log_prob( loss_con.cpu() )).exp() 
        prob_dime = (mvn4.log_prob( loss_con.cpu() )).exp() 
        prob_dimf = (mvn5.log_prob( loss_con.cpu() )).exp() 

        prob_loss_con2 = torch.cat(( \
                                    prob_dima.unsqueeze(1), prob_dimb.unsqueeze(1), prob_dimc.unsqueeze(1), \
                                    prob_dimd.unsqueeze(1), prob_dime.unsqueeze(1), prob_dimf.unsqueeze(1), \
                                    ), dim=1).float()

        prob_loss_con = prob_loss_con2 * torch.tensor(lambda_all).repeat(len(prob_dima), 1)




















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

        outputs = meta_model(inputs)
        cost = ce_loss(outputs, targets, reduce=False).reshape((-1,1))
        v_lambda = vnet(cost.detach())
        l_f_meta = ((cost * v_lambda).sum())/(v_lambda.detach().sum())
        optimizer_temp.step(l_f_meta)

        inputs_val, targets_val = next(train_meta_loader_iter)
        y_g_hat = meta_model(inputs_val)
        l_g_meta = nn.cross_entropy_loss(y_g_hat, targets_val)

        optimizer_vnet.step(l_g_meta)

        outputs = model(inputs)
        cost_w = ce_loss(outputs, targets, reduce=False).reshape((-1, 1))
        
        with jt.no_grad():
            w_new = vnet(cost_w)

        loss = ((cost_w * w_new).sum())/(w_new.detach().sum())

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
vnet = VNet(1, 100, 1)

optimizer_model = jt.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_vnet = jt.optim.Adam(vnet.parameters(), 1e-3, weight_decay=1e-4)


def main():
    best_acc = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch)
        meta_lr = print_lr(optimizer_model, epoch)
        train(train_loader,train_meta_loader,model, vnet,optimizer_model,optimizer_vnet,epoch, meta_lr)
        test_acc = test(model=model, test_loader=test_loader)
        # vnet.save('./save/vnet_%d.pkl' % epoch)
        if test_acc >= best_acc:
            best_acc = test_acc

    print('best accuracy:', best_acc)


if __name__ == '__main__':
    main()
