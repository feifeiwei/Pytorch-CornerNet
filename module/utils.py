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

def detect(heat_tl,heat_br,tag_tl,tag_br,offset_tl,offset_br,k=100,ae_threshold=0.5,num_dets=100, down_ratio=4):
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
    bboxes = _gather_feat(bboxes, idx)*down_ratio
    
    cls = cls_tl.contiguous().view(batch, -1, 1)
    cls = _gather_feat(cls,idx).float()
    
    scores_tl = scores_tl.contiguous().view(batch, -1, 1)
    scores_tl = _gather_feat(scores_tl,idx)
    
    scores_br = scores_br.contiguous().view(batch, -1, 1)
    scores_br = _gather_feat(scores_br,idx)
    
    detections = torch.cat([bboxes, scores, scores_tl, scores_br, cls],2)
    return detections


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
       B, ?, ?
    
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # box_corner = prediction.new(prediction.shape)
    
    # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    
    # prediction[:, :, :4] = box_corner[:, :, :4]
    
    output = [None] * len(prediction)
    
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold       
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
       # pdb.set_trace()
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        
        # Get score and class with highest confidence
        # Get score and class with highest confidence
        #class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        
        detections = image_pred#torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        
        
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]
              
            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            
    return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou