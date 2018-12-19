# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:22:49 2018

list img_path class xmin ymin xmax ymax xmin ymin xmax ymax .......

@author: 60236
"""
import math
import glob
import numpy as np
import torch

from PIL import Image
from .utils import draw_gaussian, gaussian_radius
from .augmentation import random_crop,random_flip,resize

class ListDataset(torch.utils.data.Dataset):
    def __init__(self,list_path, img_size=511, fmp_size=128, classes=80, train=True, transform=None):
        
        self.fmp_size  =  fmp_size
        self.transform =  transform
        self.img_size  =  img_size
        self.classes   =  classes
        self.train = train
        
        self.gaussian_rad = -1
        self.gaussian_apply = True
        
        with open(list_path,'r') as file:
            files = file.readlines()
            self.num_samples = len(files)
        
        self.img_path = []
        self.boxes = []
        self.labels = []
        
        for file in files:
            splited = file.strip().split()
            self.img_path.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
    
    def __getitem__(self,idx):
        img = Image.open(self.img_path[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.img_size
        
        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, (size,size))
        
        img = self.transform(img)
        return img, boxes, labels
    def __len__(self):
        return self.num_samples
    
    def collate_fn(self,batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x  in batch]
        labels = [x[2] for x in batch]
        batch_size = len(imgs)
        h = w = self.img_size
        max_tag_len = 100
        
        inputs = torch.zeros(batch_size, 3, h, w)
        tl_heatmaps = torch.zeros((batch_size, self.classes, self.fmp_size, self.fmp_size))
        br_heatmaps = torch.zeros((batch_size, self.classes, self.fmp_size, self.fmp_size))
        
        tl_regrs    = torch.zeros((batch_size, max_tag_len, 2))
        br_regrs    = torch.zeros((batch_size, max_tag_len, 2))
        
        tl_tags     = torch.zeros((batch_size, max_tag_len))
        br_tags     = torch.zeros((batch_size, max_tag_len))
        
        tag_masks   = torch.zeros((batch_size, max_tag_len))
        tag_lens    = torch.zeros((batch_size, ))
        
        ratio = self.fmp_size / self.img_size
        for b in range(batch_size):
            inputs[b] = imgs[b]
            cur_labels = labels[b]
            for i in range(len(boxes[b])):
                label = cur_labels[i].item()
                xtl,ytl = boxes[b][i][0], boxes[b][i][1]
                xbr,ybr = boxes[b][i][2], boxes[b][i][3]
                fxtl = (xtl * ratio)  ##
                fytl = (ytl * ratio)
                fxbr = (xbr * ratio)
                fybr = (ybr * ratio)
                
                xtl = int(fxtl)
                ytl = int(fytl)
                xbr = int(fxbr)
                ybr = int(fybr)
                if self.gaussian_apply:
                    width = boxes[b][i][2] - boxes[b][i][0]
                    height = boxes[b][i][3] - boxes[b][i][1]
                    
                    width  = math.ceil(width * ratio) ## to the upper
                    height = math.ceil(height * ratio)
                    if self.gaussian_rad == -1:
                        radius = gaussian_radius((height, width), 0.7)
                        radius = max(0, int(radius))
                    else:
                        radius = self.gaussian_rad
                    draw_gaussian(tl_heatmaps[b, label], (xtl, ytl), radius)
                    draw_gaussian(br_heatmaps[b, label], (xbr, ybr), radius)
                else:
                    tl_heatmaps[b, label, xtl, ytl] = 1
                    br_heatmaps[b, label, xbr, ybr] = 1
                    
                tag_idx = tag_lens[b].long().item()
                tl_regrs[b, tag_idx,:] = torch.Tensor([fxtl - xtl, fytl - ytl]) ##offsets
                br_regrs[b, tag_idx,:] = torch.Tensor([fxbr - xbr, fybr - ybr])
                tl_tags[b,tag_idx] = ytl * self.fmp_size + xtl
                br_tags[b,tag_idx] = ybr * self.fmp_size + xbr
                
                tag_lens[b] += 1
        for b in range(batch_size):
            tag_len = tag_lens[b].long().item()  ## nums of boxes in each images
            tag_masks[b, :tag_len] = 1
            
        return {
                "inputs":inputs,
                "targets": [tl_heatmaps, br_heatmaps, tl_tags, br_tags, tl_regrs, br_regrs, tag_masks]
                }


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, folder_path, img_size=511, transform=None):
        self.files = sorted(glob.glob(r'%s/*.*' % folder_path))  #5011 pics
        self.img_shape = (img_size, img_size)
        self.transform = transform
    def __getitem__(self,index):
        image_path = self.files[index % len(self.files)]
        #extract images
        img = np.array(Image.open(image_path))  # h w 
        h, w ,_ = img.shape
        dim_diff = np.abs(h-w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff // 2
        #Determine padding
        
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=127.5)
        # Resize and normalize
        input_img = Image.fromarray(input_img).resize(self.img_shape,Image.ANTIALIAS)
        # Channels-first
        if self.transform is not None:
            input_img = self.transform(input_img)
        else:
            input_img = np.transpose(input_img, (2, 0, 1))
            # As pytorch tensor
            input_img = torch.from_numpy(input_img).float()

        return image_path, input_img
    def __len__(self):
        return len(self.files)

                