from __future__ import print_function, division
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import collections
import torch.utils.data as data
import shutil
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity

class ADE20KDataset(Dataset):
    def __init__(self,ROOT_DIR, period, transform=None):
        self.root_dir = ROOT_DIR
        self.rst_dir = os.path.join(self.root_dir,'ADEChallengeData2016','result')
        self.period = period
        self.num_categories = 150
        self.transform = transform
        self.odgt = None        

        if self.period == 'train':
            self.odgt = os.path.join(self.root_dir,'ADEChallengeData2016','train.odgt')
        else:
            self.odgt = os.path.join(self.root_dir,'ADEChallengeData2016','validation.odgt')

        self.list_sample = [json.loads(x.rstrip()) for x in open(self.odgt, 'r')]

    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.list_sample[idx]['fpath_img'])
        img = Image.open(image_path).convert('RGB')
        r = self.list_sample[idx]['height']
        c = self.list_sample[idx]['width']

        name = self.list_sample[idx]['fpath_img'].replace('ADEChallengeData2016/images/','')
        if self.period == 'train':
            name = name.replace('train/','') 
        if 'val' in self.period:
            name = name.replace('validation/','') 
        assert(self.period != 'test')
        name = name.replace('.jpg','')
        
        sample = {'image': img, 'name': name, 'row': r, 'col': c}

        if self.period == 'train' or self.period == 'val':
            seg_path = os.path.join(self.root_dir, self.list_sample[idx]['fpath_segm'])
            seg = Image.open(seg_path)
            sample['segmentation'] = seg
            #assert(seg.ndim == 2)
            assert(img.size[0] == seg.size[0])
            assert(img.size[1] == seg.size[1])
        if self.transform is not None:
            img, target = self.transform(img, seg)

        return img, target
 
    def decode_target(self, label):
        m = label.astype(np.uint16)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3 | (m&64)>>1
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2 | (m&128)>>2
        cmap[:,:,2] = (m&4)<<5 | (m&32)<<1
        return cmap