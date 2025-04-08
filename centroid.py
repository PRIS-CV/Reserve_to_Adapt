from __future__ import division, print_function, absolute_import
import numpy as np
import torch
from utilities import *
import torch.nn.functional as F

class Centroids(object):
    def __init__(self, class_num, dim, use_cuda):
        self.class_num = class_num
        self.src_ctrs = torch.ones((class_num, dim))
        self.tgt_ctrs = torch.ones((class_num, dim+1))
        self.src_ctrs *= 1e-10
        self.tgt_ctrs *= 1e-10
        self.dim = dim
        if use_cuda:
            self.src_ctrs = self.src_ctrs.cuda()
            self.tgt_ctrs = self.tgt_ctrs.cuda()
            

    def get_centroids(self, domain=None, cid=None):
        if domain == 'source':
            return self.src_ctrs if cid is None else self.src_ctrs[cid, :]
        elif domain == 'target':
            return self.tgt_ctrs if cid is None else self.tgt_ctrs[cid, :]
        else:
            return self.src_ctrs, self.tgt_ctrs
    

    

    @torch.no_grad()
    def update(self, pred_s, pred_t, label_s,label_unk=None, ):
        self.upd_src_centroids(pred_s, label_s)
        self.upd_tgt_centroids(pred_t, label_unk)

    
    @torch.no_grad()
    def update_cweight(self, label_unk):
        c_weight = torch.zeros(self.class_num)
        if len(label_unk) == 0:
            return c_weight  # 
        for i in range(self.class_num):
            count = np.sum(label_unk == i)  # 直接使用Numpy的sum方法
            if count>0:
                c_weight[i]+=sum(label_unk==i)
        c_weight = c_weight/torch.sum(c_weight)
        return c_weight

    
    @torch.no_grad()
    def upd_src_centroids(self, probs, labels):      
        for i in range(self.class_num):
            
            data_idx = np.argwhere(labels[:,i] == 1)[:,0]
            new_centroid = torch.mean(torch.tensor(probs[data_idx, :self.dim]), 0).squeeze()
        
            self.src_ctrs[i, :] = new_centroid.cuda()
        

    @torch.no_grad()
    def upd_tgt_centroids(self, probs, labels):
        
        if labels is None:
            return
       
        for i in range(self.class_num):
            
            data_idx = np.argwhere(labels==i)
            new_centroid = torch.mean(torch.tensor(probs[data_idx]), 0).squeeze()
            self.tgt_ctrs[i, :] = new_centroid.cuda()

