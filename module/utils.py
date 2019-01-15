# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:23:07 2018

@author: 60236
"""
import torch

def comp(a,b,A,B):
    batch = a.size(0)
    a_ = a.unsqueeze(1).contiguous().view(batch,1,-1)
    b_ = b.unsqueeze(1).contiguous().view(batch,1,-1)
    c_ = torch.cat((a_,b_),1)
    m = c_.max(1)[0].unsqueeze(1).expand_as(c_)
    m = (c_==m).float()
    m1 = m.permute(0,2,1)
    k = m1[...,0]
    j = m1[...,1]
    z = ((k*j)!=1).float()
    j = z*j
    m1 = torch.cat((k,j),1).unsqueeze(1).view_as(m)

    A_ = A.unsqueeze(1).contiguous().view(batch,1,-1)
    B_ = B.unsqueeze(1).contiguous().view(batch,1,-1)
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

def _topk(scores, K=20):
    batch, _, height, width = scores.size()

    topk_scores, topk_idx = torch.topk(scores.view(batch, -1), K)

    topk_cls = (topk_idx / (height * width)).int()

    topk_idx = topk_idx % (height * width)
    topk_ys   = (topk_idx / width).int().float()
    topk_xs   = (topk_idx % width).int().float()
    return topk_scores, topk_idx, topk_cls, topk_ys, topk_xs

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def detect(heat_tl,heat_br,tag_tl,tag_br,offset_tl,offset_br,k=100,ae_threshold=0.5,num_dets=1000):
    batch, cls, height, width = heat_tl.size()##?,20,128,128
    
    heat_tl = heat_tl.sigmoid()
    heat_br = heat_br.sigmoid()

    heat_tl = _nms(heat_tl)
    heat_br = _nms(heat_br)
    
    scores_tl, idx_tl, cls_tl, ys_tl, xs_tl = _topk(heat_tl, K=k)
    scores_br, idx_br, cls_br, ys_br, xs_br = _topk(heat_br, K=k)
    
    ys_tl = ys_tl.view(batch, k, 1).expand(batch, k, k)
    xs_tl = xs_tl.view(batch, k, 1).expand(batch, k, k)
    ys_br = ys_br.view(batch, 1, k).expand(batch, k, k)
    xs_br = xs_br.view(batch, 1, k).expand(batch, k, k)
    
    offset_tl = tranpose_and_gather_feat(offset_tl, idx_tl).view(batch,k,1,2)
    offset_br = tranpose_and_gather_feat(offset_br, idx_br).view(batch,1,k,2)
    
    xs_tl = xs_tl + offset_tl[...,0]
    ys_tl = ys_tl + offset_tl[...,1]
    xs_br = xs_br + offset_br[...,0]
    ys_br = ys_br + offset_br[...,1]
    
    bboxes = torch.stack((xs_tl,ys_tl,xs_br,ys_br),dim=3) # ?, 100,100,4
    
    tag_tl = tranpose_and_gather_feat(tag_tl, idx_tl)
    tag_tl = tag_tl.view(batch, k, 1)
    tag_br = tranpose_and_gather_feat(tag_br, idx_br)
    tag_br = tag_br.view(batch, 1, k)
    dists  = torch.abs(tag_tl - tag_br) ##1,100,100
    
    scores_tl = scores_tl.view(batch, k, 1).expand(batch, k, k)
    scores_br = scores_br.view(batch, 1, k).expand(batch, k, k)
    scores    = (scores_tl + scores_br) / 2 ##1,100,100
    
    # reject boxes based on classes
    cls_tl = cls_tl.view(batch, k, 1).expand(batch, k, k)
    cls_br = cls_br.view(batch, 1, k).expand(batch, k, k)
    cls_idx = (cls_br != cls_tl) #1,100,100
    
    #reject boxes based on dist
    dists_idx = dists > ae_threshold
    
    # reject boxes based on widths and heights
    width_idx = xs_br < xs_tl
    height_idx = ys_br < ys_tl
    
    
    scores[cls_idx] = -1
    scores[dists_idx] = -1
    scores[height_idx] = -1
    scores[width_idx] = -1
    
    scores = scores.view(batch,-1)
    scores, idx = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)
    
    bboxes = bboxes.view(batch,-1,4)
    bboxes = _gather_feat(bboxes, idx)
    
    cls = cls_tl.contiguous().view(batch, -1, 1)
    cls = _gather_feat(cls,idx).float()
    
    scores_tl = scores_tl.contiguous().view(batch, -1, 1)
    scores_tl = _gather_feat(scores_tl,idx)
    
    scores_br = scores_br.contiguous().view(batch, -1, 1)
    scores_br = _gather_feat(scores_br,idx)
    
    detections = torch.cat([bboxes, scores, scores_tl, scores_br, cls],2)
    return detections


