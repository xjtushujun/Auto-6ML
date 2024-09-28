import os
import json
# import tools
import random
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
from PIL import Image
import jittor.transform as transforms


def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C

        
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset): 
    def __init__(self, data_name, root_dir, transform, mode, noise_file='', losses=[]): 
        super(cifar_dataset, self).__init__()
        self.transform = transform
        self.mode = mode  
     
        if self.mode=='test':
            if data_name=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif data_name=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if data_name=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
                num_class = 10
            elif data_name=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                train_coarse_labels = train_dic['coarse_labels']
                num_class = 100
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            noise_label = json.load(open(noise_file,"r"))

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
                self.train_label = train_label
                print('the right number is ', ( ( jt.array(self.noise_label) == jt.array(self.train_label) ).sum().float() ))

            else:
                if self.mode == "meta":
                    idx_to_meta = []

                    data_list = {}
                    for j in range(num_class):
                        data_list[j] = [i for i, label in enumerate(noise_label) if label == j]

                    for _, img_id_list in data_list.items():
                        _, indexs = jt.topk(losses[img_id_list], 10, largest=False)

                        idx_to_meta.extend(((jt.array(img_id_list))[indexs]).tolist())

                    self.train_data = [train_data[i] for i in idx_to_meta]
                    self.noise_label = [noise_label[i] for i in idx_to_meta]
                    clean_label =  [train_label[i] for i in idx_to_meta]

                    print("%s data has a size of %d" % (self.mode, len(self.train_data)))
                    print('the right number is ', ( ( jt.array(self.noise_label) == jt.array(clean_label) ).sum().float() ))
            
    def __getitem__(self, index):
        if self.mode=='meta':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target 
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, data_name, batch_size, num_workers, root_dir, noise_file=''):
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file

        normalize = transforms.ImageNormalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (np.pad( x, ((0,0), (4,4), (4,4)), mode='reflect')).transpose(1,2,0)),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

    def run(self,mode,losses=[]):
        if mode=='warmup':
            trainloader = cifar_dataset(data_name=self.data_name, root_dir=self.root_dir, transform=self.transform_train, mode="all", noise_file=self.noise_file).set_attrs(num_workers=self.num_workers, batch_size=self.batch_size,shuffle=True)                       
            return trainloader
                                     
        elif mode=='test':
            test_loader = cifar_dataset(data_name=self.data_name, root_dir=self.root_dir, transform=self.transform_test, mode='test').set_attrs(num_workers=4, batch_size=self.batch_size,shuffle=False)        
            return test_loader
        
        elif mode=='eval_train':
            eval_loader = cifar_dataset(data_name=self.data_name, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file).set_attrs(num_workers=self.num_workers, batch_size=self.batch_size,shuffle=False)         
            return eval_loader  

        elif mode == 'meta':
            meta_trainloader = cifar_dataset(data_name=self.data_name, root_dir=self.root_dir, transform=self.transform_train, mode="meta",noise_file=self.noise_file, losses=losses).set_attrs(num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)  

            return meta_trainloader
      
