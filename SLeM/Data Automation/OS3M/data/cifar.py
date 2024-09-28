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
import jittor.transform as transform


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

   
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


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
                 download=False, seed=1, noisy_dataset='cifar10', on=0., noise_data_dir='./'):

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
                train_data = self.train_data[idx_to_train]
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




                if noisy_dataset == 'imagenet32':
                    noise_data = []
                    _train_list = ['train_data_batch_1',
                                'train_data_batch_2',
                                'train_data_batch_3',
                                'train_data_batch_4',
                                'train_data_batch_5',
                                'train_data_batch_6',
                                'train_data_batch_7',
                                'train_data_batch_8',
                                'train_data_batch_9',
                                'train_data_batch_10']                

                    for f in _train_list:
                        file_imagenet32 = os.path.join(noise_data_dir, f)

                        with open(file_imagenet32, 'rb') as fo:
                            entry = pickle.load(fo, encoding='latin1')
                            noise_data.append(entry['data'])
                    noise_data = np.concatenate(noise_data)
                    noise_data = noise_data.reshape((-1, 3, 32, 32))
                    noise_data = noise_data.transpose((0, 2, 3, 1))  # Convert to HWC

                    # print('imagenet32 data:', noise_data.shape)

                elif noisy_dataset == 'cifar100':
                    noise_data = unpickle('%s/train'%noise_data_dir)['data'].reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
                elif noisy_dataset == 'cifar10':
                    noise_data=[]
                    for n in range(1,6):
                        dpath = '%s/data_batch_%d'%(noise_data_dir,n)
                        data_dic = unpickle(dpath)
                        noise_data.append(data_dic['data'])
                    noise_data = ( np.concatenate(noise_data) ).reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
                elif noisy_dataset == 'SVHN':
                    noise_data = ( torchvision.datasets.SVHN(root='./data', split='train', transform=None, download=True) ).data.transpose((0, 2, 3, 1))
            
                print('start product noising...................................')

                #inject noise   
                idx = list(range(train_data.shape[0]))                # indices of cifar dataset
                random.shuffle(idx)                 
                num_open_noise = int(on*train_data.shape[0])     # total amount of noisy/openset images
                if noisy_dataset == 'imagenet32':       # indices of openset source images
                    target_noise_idx = list(range(1281149))
                elif noisy_dataset == 'SVHN':
                    target_noise_idx = list(range(50000))
                else:
                    target_noise_idx = list(range(train_data.shape[0]))
                random.shuffle(target_noise_idx)  
                self.open_noise = list(zip(idx[:num_open_noise], target_noise_idx[:num_open_noise]))  # clean sample -> openset sample mapping
                                
                print('start product img...................................')

                if os.path.exists( 'noise_data_%s_%s_%.1f_%.2f_%s.npy' % (dataset, corruption_type, corruption_prob, on, noisy_dataset) ):
                    train_data = np.load( 'noise_data_%s_%s_%.1f_%.2f_%s.npy' % (dataset, corruption_type, corruption_prob, on, noisy_dataset) )
                else:
                    for cleanIdx, noisyIdx in self.open_noise:
                        if noisy_dataset == 'imagenet32':
                            # train_data[cleanIdx] = np.asarray(Image.open('{}/{}.png'.format(noise_data_dir, str(noisyIdx+1).zfill(7)))).reshape((32,32,3))
                            train_data[cleanIdx] = noise_data[noisyIdx]
                        elif noisy_dataset == 'cifar100':
                            train_data[cleanIdx] = noise_data[noisyIdx]
                        elif noisy_dataset == 'cifar10':
                            train_data[cleanIdx] = noise_data[noisyIdx]
                        elif noisy_dataset == 'SVHN':
                            train_data[cleanIdx] = noise_data[noisyIdx]
                        elif noisy_dataset == 'gaussian':
                            # print ("OOD: Gaussian")
                            print('cleanIdx:', cleanIdx)
                            noise = np.random.normal(0.2, 1, (32, 32, 3))
                            img = train_data[cleanIdx] + 255*noise
                            img = np.clip(img, 0, 255).astype(np.uint8)
                            train_data[cleanIdx] = img
                        elif noisy_dataset == 'square':
                            # print ("OOD: Square")
                            print('cleanIdx:', cleanIdx)
                            train_data[cleanIdx][2:30, 2:30, :] = 0 if np.random.rand(1) < 0.5 else 1
                            # torch.save(train_data, 'noise_data_%s_%s_%s_%s.pth' % (dataset, self.r, self.on, noisy_dataset) )
                        elif noisy_dataset == 'reso':
                            # print ("OOD: Resolution")
                            print('cleanIdx:', cleanIdx)
                            img = Image.fromarray(train_data[cleanIdx])
                            img = img.resize((4,4)).resize((32,32))
                            train_data[cleanIdx] = np.array(img)
                            # torch.save(train_data, 'noise_data_%s_%s_%s_%s.pth' % (dataset, self.r, self.on, noisy_dataset) )
                        elif noisy_dataset == 'gaussian1':
                            # print ("OOD: Gaussian")
                            print('cleanIdx:', cleanIdx)
                            noise = np.random.normal(0.2, 1, (32, 32, 3))
                            train_data[cleanIdx] = np.clip(noise, 0., 1.) * 255
                        elif noisy_dataset == 'cutout':
                            train_data[cleanIdx][8:24, 8:24, :] = 0.
                        elif noisy_dataset=='gaussian_blur':
                            train_data[cleanIdx] = np.clip( gaussian(np.array(train_data[cleanIdx]) / 255., sigma=10, multichannel=True), 0., 1. ) * 255
                
                self.train_data = train_data

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

        return img, target, index

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


def build_dataset(args):
    if args.dataset == 'cifar10':
        normalize = transform.ImageNormalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std =[x / 255.0 for x in [63.0, 62.1, 66.7]])
        train_transform = transform.Compose([
            transform.ToTensor(),
            transform.Lambda(lambda x: (np.pad( x, ((0,0), (4,4), (4,4)), mode='reflect')).transpose(1,2,0)),
            transform.ToPILImage(),
            transform.RandomCrop(32),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            normalize,
        ])
        test_transform = transform.Compose([
            transform.ToTensor(),
            normalize
        ])

        train_meta_loader = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, noisy_dataset=args.noisy_dataset, on=args.on, noise_data_dir=args.noise_data_dir).set_attrs(num_workers=args.prefetch, batch_size=args.batch_size,shuffle=True)
        train_loader = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed, noisy_dataset=args.noisy_dataset, on=args.on, noise_data_dir=args.noise_data_dir).set_attrs(num_workers=args.prefetch, batch_size=args.batch_size,shuffle=True)
        test_loader = CIFAR10(root='../data', train=False, transform=test_transform, download=True).set_attrs(num_workers=args.prefetch, batch_size=args.batch_size,shuffle=False)


    elif args.dataset == 'cifar100':
        normalize = transform.ImageNormalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        train_transform = transform.Compose([
            transform.ToTensor(),
            transform.Lambda(lambda x: (np.pad( x, ((0,0), (4,4), (4,4)), mode='reflect')).transpose(1,2,0)),
            transform.ToPILImage(),
            transform.RandomCrop(32),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            normalize,
        ])
        test_transform = transform.Compose([
            transform.ToTensor(),
            normalize
        ])

        train_meta_loader = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, noisy_dataset=args.noisy_dataset, on=args.on, noise_data_dir=args.noise_data_dir).set_attrs(num_workers=args.prefetch, batch_size=args.batch_size,shuffle=True)
        train_loader = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed, noisy_dataset=args.noisy_dataset, on=args.on, noise_data_dir=args.noise_data_dir).set_attrs(num_workers=args.prefetch, batch_size=args.batch_size,shuffle=True)
        test_loader = CIFAR100(root='../data', train=False, transform=test_transform, download=True).set_attrs(num_workers=args.prefetch, batch_size=args.batch_size,shuffle=True)

    return train_loader, train_meta_loader, test_loader



