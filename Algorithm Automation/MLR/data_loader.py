import gzip
import tarfile
import zipfile
from jittor_utils.misc import download_url_to_local, check_md5
from PIL import Image
import sys
import pickle
import numpy as np
import os
from jittor.dataset import Dataset
import jittor.transform as transforms


def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)


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


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url_to_local(url, filename, download_root, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


class CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    Example::
        from jittor.dataset.cifar import CIFAR10
        a = CIFAR10()
        a.set_attrs(batch_size=16)
        for imgs, labels in a:
            print(imgs.shape, labels.shape)
            break
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root='', train=True, meta=True, num_meta=1000,
                 corruption_prob=0, corruption_type='unif', transform=None, target_transform=None,
                 download=False, seed=1):

        super(CIFAR10, self).__init__()
        self.root = root
        self.transform=transform
        self.target_transform=target_transform
        self.train = train  # training set or test set
        self.meta = meta
        self.corruption_prob = corruption_prob
        self.num_meta = num_meta


        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_coarse_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                    img_num_list = [int(self.num_meta/10)] * 10
                    num_classes = 10
                else:
                    self.train_labels += entry['fine_labels']
                    self.train_coarse_labels += entry['coarse_labels']
                    img_num_list = [int(self.num_meta/100)] * 100
                    num_classes = 100
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))   # convert to HWC

            data_list_val = {}
            for j in range(num_classes):
                data_list_val[j] = [i for i, label in enumerate(self.train_labels) if label == j]

            idx_to_meta = []
            idx_to_train = []
            print(img_num_list)

            np.random.seed(seed)
            for cls_idx, img_id_list in data_list_val.items():
                np.random.shuffle(img_id_list)
                img_num = img_num_list[int(cls_idx)]
                idx_to_meta.extend(img_id_list[:img_num])
                idx_to_train.extend(img_id_list[img_num:])


            if meta is True:
                self.train_data = self.train_data[idx_to_meta]
                self.train_labels = list(np.array(self.train_labels)[idx_to_meta])
            else:
                self.train_data = self.train_data[idx_to_train]
                self.train_labels = list(np.array(self.train_labels)[idx_to_train])
                if corruption_type == 'hierarchical':
                    self.train_coarse_labels = list(np.array(self.train_coarse_labels)[idx_to_meta])

                if corruption_type == 'unif':
                    C = uniform_mix_C(self.corruption_prob, num_classes)
                    print(C)
                    self.C = C
                elif corruption_type == 'flip':
                    C = flip_labels_C(self.corruption_prob, num_classes)
                    print(C)
                    self.C = C
                elif corruption_type == 'flip2':
                    C = flip_labels_C_two(self.corruption_prob, num_classes)
                    print(C)
                    self.C = C
                elif corruption_type == 'hierarchical':
                    assert num_classes == 100, 'You must use CIFAR-100 with the hierarchical corruption.'
                    coarse_fine = []
                    for i in range(20):
                        coarse_fine.append(set())
                    for i in range(len(self.train_labels)):
                        coarse_fine[self.train_coarse_labels[i]].add(self.train_labels[i])
                    for i in range(20):
                        coarse_fine[i] = list(coarse_fine[i])

                    C = np.eye(num_classes) * (1 - corruption_prob)

                    for i in range(20):
                        tmp = np.copy(coarse_fine[i])
                        for j in range(len(tmp)):
                            tmp2 = np.delete(np.copy(tmp), j)
                            C[tmp[j], tmp2] += corruption_prob * 1/len(tmp2)
                    self.C = C
                    print(C)
                else:
                    assert False, "Invalid corruption type '{}' given. Must be in {'unif', 'flip', 'hierarchical'}".format(corruption_type)

                np.random.seed(seed)
                for i in range(len(self.train_labels)):
                    self.train_labels[i] = np.random.choice(num_classes, p=C[self.train_labels[i]])
                self.corruption_matrix = C

        else:
            f = self.test_list[0][0]
            file = os.path.join(root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        # print('img:', img.size)
        # a = transform.ToTensor()(img)
        # print('shape:', a.shape)
        # tt = np.pad(a, ((0,0),(4,4),(4,4)))
        # tt = transform.ToPILImage()(tt)
        # print('tt:',tt.size)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            if self.meta is True:
                return self.num_meta
            else:
                return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    Example::
        from jittor.dataset.cifar import CIFAR100
        a = CIFAR100()
        a.set_attrs(batch_size=16)
        for imgs, labels in a:
            print(imgs.shape, labels.shape)
            break
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


def build_dataset(dataset,num_meta,batch_size):

    if dataset == 'cifar10':
        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: (np.pad( x, ((0,0), (4,4), (4,4)), mode='reflect')).transpose(1,2,0)),
             transforms.ToPILImage(), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std =[0.2023, 0.1994, 0.2010]), ])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std =[0.2023, 0.1994, 0.2010]), ])
        train_data_meta = CIFAR10(
            root='../../data/', train=True, meta=True, num_meta=num_meta, corruption_prob=0,
            corruption_type='unif', transform=transform_train, download=True).set_attrs(num_workers=0, batch_size=batch_size,shuffle=True)
        train_data = CIFAR10(
            root='../../data/', train=True, meta=False, num_meta=num_meta, corruption_prob=0,
            corruption_type='unif', transform=transform_train, download=True).set_attrs(num_workers=0, batch_size=batch_size,shuffle=True)
        test_data = CIFAR10(root='../../data/', train=False, transform=transform_test, download=True).set_attrs(num_workers=0, batch_size=batch_size,shuffle=False)


    elif dataset == 'cifar100':
        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: (np.pad( x, ((0,0), (4,4), (4,4)), mode='reflect')).transpose(1,2,0)),
             transforms.ToPILImage(), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.ImageNormalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]), ])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.ImageNormalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]), ])
        train_data_meta = CIFAR100(
            root='../../data/', train=True, meta=True, num_meta=num_meta, corruption_prob=0,
            corruption_type='unif', transform=transform_train, download=True).set_attrs(num_workers=0, batch_size=batch_size,shuffle=True)
        train_data = CIFAR100(
            root='../../data/', train=True, meta=False, num_meta=num_meta, corruption_prob=0,
            corruption_type='unif', transform=transform_train, download=True).set_attrs(num_workers=0, batch_size=batch_size,shuffle=True)
        test_data = CIFAR100(root='../../data/', train=False, transform=transform_test, download=True).set_attrs(num_workers=0, batch_size=batch_size,shuffle=False)

    return train_data, train_data_meta, test_data
