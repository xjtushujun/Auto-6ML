# -*- coding: utf-8 -*-
import random
import torch
import numpy as np
from torch.utils.data.sampler import Sampler

class RandomCycleIter:
    """Randomly iterate element in each cycle

    Example:
        >>> rand_cyc_iter = RandomCycleIter([1, 2, 3])
        >>> [next(rand_cyc_iter) for _ in range(10)]
        [2, 1, 3, 2, 3, 1, 1, 2, 3, 2]
    """
    def __init__(self, data):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            random.shuffle(self.data_list)
        return self.data_list[self.i]

    next = __next__  # Py2


def class_aware_sample_generator(cls_iter, data_iter_list, n):
    i = 0
    while i < n:
        yield next(data_iter_list[next(cls_iter)])
        i += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_samples_cls=0):
        self.data_source = data_source
        n_cls = len(np.unique(data_source.targets))
        self.class_iter = RandomCycleIter(range(n_cls))
        cls_data_list = [list() for _ in range(n_cls)]
        for i, label in enumerate(data_source.targets):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        if num_samples_cls:
            self.num_samples = num_samples_cls * len(cls_data_list)
        else:
            self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list, self.num_samples)

    def __len__(self):
        return self.num_samples
    
def get_sampler():
    return ClassAwareSampler