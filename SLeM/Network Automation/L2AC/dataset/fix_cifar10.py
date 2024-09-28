import numpy as np
from PIL import Image
import jittor as jt
# import torchvision
import torch
# from torchvision.transforms import transforms
import jittor.transform as transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

# Parameters for data
cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

# Augmentations.
# transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(cifar10_mean, cifar10_std)
#     ])
normalize = transforms.ImageNormalize(mean=cifar10_mean, std=cifar10_std)
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (np.pad( x, ((0,0), (4,4), (4,4)), mode='reflect')).transpose(1,2,0)),
    transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_strong = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (np.pad( x, ((0,0), (4,4), (4,4)), mode='reflect')).transpose(1,2,0)),
    transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
# transform_strong = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(cifar10_mean, cifar10_std)
# ])
# transform_strong.transforms.insert(0, RandAugment(3, 4))
# transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.ImageNormalize(cifar10_mean, cifar10_std)
])

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3


def get_cifar10(root, l_samples, u_samples, transform_train=transform_train, transform_strong=transform_strong,
                transform_val=transform_val, download=True):
    base_dataset = jt.dataset.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples)
    print('train_labeled_idxs, train_unlabeled_idxs:', len(train_labeled_idxs), len(train_unlabeled_idxs) )

    train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)

    train_labeled_dataset1 = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)

    print('transform_strong:', transform_strong, transform_strong.__dict__ )
    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True,
                                                transform=TransformTwice(transform_train, transform_strong))
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=False)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_labeled_dataset1

def train_split(labels, n_labeled_per_class, n_unlabeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])

    return train_labeled_idxs, train_unlabeled_idxs

class CIFAR10_labeled(jt.dataset.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]
        print('self.data', len(self.data), len(self.targets), indexs is not None )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # print('index:', index)
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def __len__(self):
        return len(self.data)

    def get_cls_num_list(self):
    
        n_cls = len(np.unique(self.targets))
        cls_num_list = [0]*n_cls
        
        for label in self.targets:
            cls_num_list[int(label)] += 1

        return cls_num_list
    

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
