# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:23:07 2018

@author: 60236
"""

import torch

def comp(a,b,A,B):
    batch = a.size(0)
    a_ = a.unsqueeze(1).view(batch,1,-1)
    b_ = b.unsqueeze(1).view(batch,1,-1)
    c_ = torch.cat((a_,b_),1)
    m = c_.max(1)[0].unsqueeze(1).expand_as(c_)
    m = (c_==m).float()
    m1 = m.permute(0,2,1)
    k = m1[...,0]
    j = m1[...,1]
    z = ((k*j)!=1).float()
    j = z*j
    m1 = torch.cat((k,j),1).unsqueeze(1).view_as(m)

    A_ = A.unsqueeze(1).view(batch,1,-1)
    B_ = B.unsqueeze(1).view(batch,1,-1)
    C_ = torch.cat((A_,B_),1).permute(0,2,1)
    m1 = m1.long().permute(0,2,1)
    res = C_[m1.long()==1].view_as(a)

    return res

def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat