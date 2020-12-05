from __future__ import print_function, division
import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import tarfile
import pickle
import collections
import torch.utils.data as data
import shutil
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
import pickle
class LvisDataset(Dataset):
    with open('LVIS_COCO/color_map_lvis.pkl','rb') as f:
        lvis_cmap = pickle.load(f) 
    inds = sorted(list(lvis_cmap.keys()))
    train_id_to_color = []
    for ind in inds:
        train_id_to_color.append(lvis_cmap[ind])
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    
    def __init__(self,ROOT_DIR, period, transform=None):
        self.root_dir = ROOT_DIR
        self.rst_dir = os.path.join(self.root_dir,'lvisdataset')
        self.period = period
        self.num_categories = 1284
        self.transform = transform
        self.odgt = None
        if self.period == 'train':
            self.imid2path = pickle.load(open(os.path.join(self.root_dir,'LVIS_COCO/img2info.pkl'),'rb'))
            del self.imid2path[429995]
        else:
            self.imid2path = pickle.load(open(os.path.join(self.root_dir,'LVIS_COCO/img2info_val.pkl'),'rb'))
        self.img_ids = list(self.imid2path.keys())

    def __len__(self):
        return len(self.imid2path.keys())

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir,'lvisdataset',self.imid2path[self.img_ids[idx]]['path'])
        img = Image.open(image_path).convert('RGB')
        r = self.imid2path[self.img_ids[idx]]['height']
        c = self.imid2path[self.img_ids[idx]]['width']
        name = self.imid2path[self.img_ids[idx]]['path']
        if self.period == 'train':
            name = name.replace('train2017/','') 
        if 'val' in self.period:
            name = name.replace('val2017/','') 
        name = self.imid2path[self.img_ids[idx]]['path']
        assert(self.period != 'test')
        name = name.replace('.jpg','')
        sample = {'image': img, 'name': name, 'row': r, 'col': c}

        if self.period == 'train' or self.period == 'val':
            seg_path = os.path.join(self.root_dir, 'lvisdataset/lvis_mask',str(self.img_ids[idx])+'.png')
            seg = Image.open(seg_path)
            sample['segmentation'] = seg
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


    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == -1] = 1284
        return cls.train_id_to_color[target]
   