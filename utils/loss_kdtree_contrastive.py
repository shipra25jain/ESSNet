import faiss
import faiss.contrib.torch_utils
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import unittest
import torch
import gc


class NNCrossEntropy(nn.Module):
    def __init__(self,ignore_index,reduction,num_neighbours,temp,dataset):
        super(NNCrossEntropy, self).__init__()
        self.dataset = dataset
        self.ignore_index = ignore_index
        self.size_average = True
        self.num_neighbours = num_neighbours
        self.temp = temp
        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()

    def forward(self, inputs, targets,class_emb):
        n_classes = class_emb.shape[0]
        neg_samples = self.num_neighbours
        if(self.dataset.lower() in ['coco','cityscapes','voc']):
            targets[targets==255]=-1
        with torch.no_grad():
            gpu_index = faiss.GpuIndexFlatL2(self.res,class_emb.shape[1])
            gpu_index.add(class_emb)
            trans_inputs = torch.transpose(torch.transpose(inputs,1,2),2,3).reshape(inputs.size()[0]*inputs.size()[2]*inputs.size()[3],class_emb.shape[1])
            D, I = gpu_index.search(trans_inputs, neg_samples)
            ret_index = torch.transpose(torch.transpose(I.reshape(inputs.size()[0],inputs.size()[2],inputs.size()[3],neg_samples),2,3),1,2)
            mask_tar = (targets.unsqueeze(1) == ret_index)
            ret_index[mask_tar] = ret_index[mask_tar] - 1
            ret_index[ret_index==-1] = 0 
            input_index = torch.cat([targets.unsqueeze(1),ret_index],dim=1)

        embmat = class_emb[input_index]
        embmat = torch.transpose(torch.transpose(embmat,3,4),2,3)
        dist_mat = torch.cdist(class_emb, class_emb, p=2)
        min_dist =  dist_mat.topk(2,largest=False)[0][:,1]
        reg_loss = torch.max(0.2 - min_dist, torch.zeros(min_dist.shape).cuda()).sum()/n_classes
        norm_loss = torch.norm(inputs.unsqueeze(1) - embmat,dim = 2)
        target_mod = (targets==-1)
        new_targets = -1*target_mod
        cross_entropy_loss = F.cross_entropy(-self.temp*norm_loss,new_targets,size_average=True,ignore_index=-1,reduce=True,reduction='mean')
        return cross_entropy_loss + reg_loss
