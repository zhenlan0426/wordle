#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:30:05 2022

@author: will
"""
import torch.nn as nn
from torch.nn import Embedding,Sequential, Linear, BatchNorm1d,Dropout,LeakyReLU
from torch_scatter import segment_csr
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
import time
import copy
from torch.nn.utils import clip_grad_value_

# =============================================================================
#  Data
# =============================================================================

class CustomDataset(Dataset):
    def __init__(self, ys,length,indexes,words_embed):
        self.ys = ys
        self.length = length
        self.words_embed = words_embed
        self.indexes = indexes

    def __len__(self):
        return len(self.length)

    def __getitem__(self, idx):
        if self.ys is None:
            return self.words_embed[np.array(self.indexes[idx])], self.length[idx]
        else:
            return self.words_embed[np.array(self.indexes[idx])], self.length[idx] ,self.ys[idx]
    
def collate(batch):
    data = list(zip(*batch))
    if len(data) == 3:
        words,length,ys = data
        return torch.tensor(np.concatenate(words)).long(),torch.tensor(length),torch.tensor(ys)
    else:
        words,length = data
        return torch.tensor(np.concatenate(words)).long(),torch.tensor(length)

# =============================================================================
#  Model
# =============================================================================
def MLP(in_d,out_d,multiple_factor=1,dropout=0.1):
    return Sequential(
                        Linear(in_d,in_d*multiple_factor),
                        LeakyReLU(inplace=True),
                        Dropout(dropout),
                        #BatchNorm1d(in_d*multiple_factor),
                        Linear(in_d*multiple_factor,out_d),
                        LeakyReLU(inplace=True),
                        Dropout(dropout),
                        #BatchNorm1d(out_d)
                        )

class pointNet_block(torch.nn.Module):
    def __init__(self,d,agg,dropout=0.1,multiple_factor=2):
        super(pointNet_block, self).__init__()   
        self.point_update =  MLP(d,d,multiple_factor,dropout)    
        self.group_update = MLP(d,d,multiple_factor,dropout)
        self.combine = Sequential(
                        Linear(d*2,d),
                        LeakyReLU(inplace=True),
                        Dropout(dropout),
                        #BatchNorm1d(d)
                        )
        self.agg = agg

    def forward(self, in_,seg2all,all2seg):
        out = self.point_update(in_)
        out_group = segment_csr(out,all2seg,reduce=self.agg)
        out_group = self.group_update(out_group)
        out = torch.cat([out,out_group[seg2all]],1)
        return self.combine(out)+in_

class pointNet(torch.nn.Module):
    def __init__(self,layers,embed_size,agg,dropout=0.1,multiple_factor=2):
        super(pointNet, self).__init__()
        self.embed = Embedding(26, embed_size)
        self.d = embed_size*5
        self.agg = agg
        self.mainNN = nn.ModuleList([pointNet_block(self.d,agg,dropout,multiple_factor) for _ in range(layers)])
        self.out_linear = MLP(self.d,1,multiple_factor)
        self.w = nn.parameter.Parameter(torch.tensor(0.13663686,device='cuda'))
        #self.min_ = torch.tensor(1,device='cuda')
        
    def forward(self,data):
        if len(data) == 3:
            IsTrain = True
            words,length,ys = data
        else:
            IsTrain = False
            words,length = data
            
        length_int = length.to(torch.long)
        seg2all = torch.cat([torch.ones(l,dtype=torch.long,device=length_int.device)*i for i,l in enumerate(length_int)])
        all2seg = torch.cumsum(torch.cat([torch.tensor([0],device=length_int.device),length_int]),0)
        out = self.embed(words).reshape(-1,self.d)
        for model in self.mainNN:
            out = model(out,seg2all,all2seg)
        out = self.out_linear(segment_csr(out,all2seg,reduce=self.agg))
        #out = torch.maximum(self.min_,out.squeeze() + self.w * torch.log2(length))
        out = out.squeeze() + self.w * torch.log2(length)
        if IsTrain:
            return nn.functional.mse_loss(out, ys)
        else:
            return out


class pointNetQ(torch.nn.Module):
    def __init__(self,layers,embed_size,agg,dropout=0.1,multiple_factor=2):
        super(pointNetQ, self).__init__()
        self.embed = Embedding(26, embed_size)
        self.d = embed_size*5
        self.agg = agg
        self.mainNN = nn.ModuleList([pointNet_block(self.d,agg,dropout,multiple_factor) for _ in range(layers)])
        self.out_linear = MLP(self.d,1,multiple_factor)
        self.w = nn.parameter.Parameter(torch.tensor(0.13663686,device='cuda'))
        #self.min_ = torch.tensor(1,device='cuda')
        
    def forward(self,data):
        if len(data) == 4:
            IsTrain = True
            words,length,action,ys = data
        else:
            IsTrain = False
            words,length,action = data
            
        length_int = length.to(torch.long)
        seg2all = torch.cat([torch.ones(l,dtype=torch.long,device=length_int.device)*i for i,l in enumerate(length_int)])
        all2seg = torch.cumsum(torch.cat([torch.tensor([0],device=length_int.device),length_int]),0)
        action = self.embed(action).reshape(-1,self.d)
        out = action[seg2all] + self.embed(words).reshape(-1,self.d)
        for model in self.mainNN:
            out = model(out,seg2all,all2seg)
        out = self.out_linear(segment_csr(out,all2seg,reduce=self.agg))
        #out = torch.maximum(self.min_,out.squeeze() + self.w * torch.log2(length))
        out = out.squeeze() + self.w * torch.log2(length)
        if IsTrain:
            return nn.functional.mse_loss(out, ys)
        else:
            return out
        
def train(opt,model,epochs,train_dl,val_dl,paras,clip,verbose=True,save=True):
    since = time.time()
    lossBest = 1e6
    opt.zero_grad()
    for epoch in range(epochs):
        # training #
        model.train()
        train_loss = 0
        val_loss = 0
        
        for i,data in enumerate(train_dl):
            data = [i.to('cuda') for i in data]
            loss = model(data)
            loss.backward()
            clip_grad_value_(paras,clip)
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
            
        # evaluating #
        model.eval()
        with torch.no_grad():
            for j,data in enumerate(val_dl):
                data = [i.to('cuda') for i in data]
                loss = model(data)
                val_loss += loss.item()
        
        # save model
        if val_loss<lossBest:
            lossBest = val_loss
            if save: bestWeight = copy.deepcopy(model.state_dict())   
        if verbose:
            print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f} \n'.format(epoch,train_loss/i,val_loss/j))

    
    if save: model.load_state_dict(bestWeight)
    time_elapsed = time.time() - since
    if verbose: print('Training completed in {}s'.format(time_elapsed))
    return model,lossBest