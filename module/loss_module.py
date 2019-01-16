# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:10:05 2018

@author: 60236
"""
import torch
import numpy as np
import torch.nn as nn
from .utils import  tranpose_and_gather_feat


class mul_task_loss(object):
    
    def __init__(self,pull_weight=0.1, push_weight=0.1, regr_weight=1,):
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        
    def regr_loss(self, regr, gt_regr, mask):
        num  = mask.float().sum()*2 
        mask = mask.unsqueeze(2).expand_as(gt_regr) 

        regr    = regr[mask==1]
        gt_regr = gt_regr[mask==1]
        regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss
    
    def focalloss(self, preds, gt): 
        pos_mask = gt.eq(1)
        neg_mask = gt.lt(1)
        
        loss = 0
        for i,pred in enumerate(preds):
            neg_weights = torch.pow(1 - gt[i][neg_mask[i]], 4) 
            pos_pred = pred[pos_mask[i]==1.]  
            neg_pred = pred[neg_mask[i]==1.] 
    
            pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
            
            neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights
            num_pos  = pos_mask[i].float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()
         
            if pos_pred.nelement() == 0:  #no element
                loss = loss - neg_loss
            else:
                loss = loss - (pos_loss + neg_loss) /(num_pos)
        return loss


    def ae_loss(self, tag0, tag1, masks):
        num  = masks.sum(dim=1, keepdim=True).unsqueeze(1).expand_as(tag0)

        masks = masks.unsqueeze(2)
        tag_mean = (tag0 + tag1) / 2 
        tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
        tag0 = (tag0*masks).sum()
        tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
        tag1 = (tag1*masks).sum()
        pull = tag0 + tag1
        mask = masks.unsqueeze(1) + masks.unsqueeze(2)
        mask = mask.eq(2)
        num  = num.unsqueeze(2).expand_as(mask)
        
        num2 = (num - 1) * num
      	m = 2
      	
        dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
        dist = m - torch.abs(dist)
        dist = nn.functional.relu(dist, inplace=True)
        dist = dist - m / (num + 1e-4)
        dist = dist / (num2 + 1e-4)  
        dist = dist[mask]
        push = dist.sum()
        return pull, push
    
    def __call__(self, outpus, targets):
        masks = targets[-1]   
    
        tl_tags = targets[2].long()
        br_tags = targets[3].long()
        
        ## heatmaps
        heat_maps_tl = outpus[0]
        heat_maps_br = outpus[1]
        heat_maps_tl_gt = targets[0]
        heat_maps_br_gt = targets[1]
        heat_maps_tl = heat_maps_tl.sigmoid()
        heat_maps_br = heat_maps_br.sigmoid()
    
        
        focalloss = (self.focalloss(heat_maps_tl,heat_maps_tl_gt) + \
                    self.focalloss(heat_maps_br,heat_maps_br_gt)) * 0.5
        
   
                     
         ######offsets
        offsets_tl = outpus[4] 
        offsets_br = outpus[5]
        offsets_tl = tranpose_and_gather_feat(outpus[4], tl_tags) 
        offsets_br = tranpose_and_gather_feat(outpus[5], br_tags)
        offsets_tl_gt = targets[4] 
        offsets_br_gt = targets[5]
        offsets_loss = self.regr_loss(offsets_tl,offsets_tl_gt,masks)*self.regr_weight + \
                        self.regr_loss(offsets_br,offsets_br_gt,masks)*self.regr_weight
                        
        ####embeddings
        embeddings_tl = outpus[2] 
        embeddings_br = outpus[3]
        tags_tl = tranpose_and_gather_feat(embeddings_tl,tl_tags) 
        tags_br = tranpose_and_gather_feat(embeddings_br,br_tags) 
        # tag loss
        pull_loss, push_loss = self.ae_loss(tags_tl,tags_br,masks)
          

        loss = (focalloss + pull_loss + push_loss + offsets_loss) / len(heat_maps_tl)
        
        return loss, [focalloss.item(), offsets_loss.item(), pull_loss.item(), push_loss.item()]
