import itertools
import numpy as np
import math
import pickle
from collections import defaultdict
#from typing import Optional
import torch
from torch.utils.data.sampler import Sampler
import os
#from detectron2.utils import comm
import json
from PIL import Image
def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = [ints]
    return all_ints[0]

class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but suitable for training on class imbalanced datasets
    like LVIS. In each epoch, an image may appear multiple times based on its "repeat
    factor". The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1] is defined
    as the fraction of images in the training set (without repeats) in which category c
    appears.
    See :paper:`lvis` (>= v2) Appendix B.2.
    """

    def __init__(self, repeat_thresh, shuffle=True, seed=None):
        """
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = shared_random_seed()
        self._seed = int(seed)

        #self._rank = comm.get_rank()
        #self._world_size = comm.get_world_size()

        # Get fractional repeat factors and split into whole number (_int_part)
        # and fractional (_frac_part) parts.
        #/scratch_net/knurrhahn/knurrhahn/shijain/ade_deeplab/DeepLabV3Plus-Pytorch
        with open('/scratch_net/knurrhahn/knurrhahn/shijain/ade_deeplab/DeepLabV3Plus-Pytorch/repeatfactors.pkl', "rb") as f:
           rep_factors =  pickle.load(f)
        rep_factors = torch.tensor(rep_factors, dtype=torch.float32)
        #rep_factors = self._get_repeat_factors(repeat_thresh)
        self._int_part = torch.trunc(rep_factors)
        self._frac_part = rep_factors - self._int_part

    def _get_repeat_factors(self, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors.
        Args:
            See __init__.
        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        odgt = os.path.join('/scratch_net/knurrhahn/knurrhahn/shijain/ade_deeplab/DeepLabV3Plus-Pytorch/ade20k/data','ADEChallengeData2016','train.odgt')
        list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
    #data_dicts = []
        category_freq = defaultdict(int)
        unique_classes = []  
        for i in range(len(list_sample)):
            seg_path = os.path.join('/scratch_net/knurrhahn/knurrhahn/shijain/ade_deeplab/DeepLabV3Plus-Pytorch/ade20k/data', list_sample[i]['fpath_segm'])
            seg = Image.open(seg_path)
            ann=list(seg.getdata())
            ann  = [a-1 for a in ann]
            unique_classes.append(list(set(ann)))  
            for cat_id in ann:
                if(cat_id!=-1):
                    category_freq[cat_id] += 1
        num_images = len(list_sample)
        print("done first round")
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }
        category_rep[-1]=0
        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for cat_ids in unique_classes:
            #cat_ids = {ann["category_id"] for ann in dataset_dict["annot
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.
        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.
        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        print("length of indices : ", len(indices))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        start =  0
        yield from itertools.islice(self._infinite_indices(), start, None, 1)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm]
            else:
                yield from indices


