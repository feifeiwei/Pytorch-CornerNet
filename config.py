# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:34:46 2018

@author: 60236
"""
config = {
        'image_size':416,
        'fms_size':104,
        
        'pull_weight':0.1, 
        'push_weight':0.1, 
        'offset_weight':1,    
        
        'num_classes':2,
        'classes':['car',],
        'epochs':200,
        
        'save_dir':r'./checkpoint/ckpt.pth',
        
        'root': r"E:\datasets\UCAS\images",
        'train_root': r'E:\datasets\UCAS\train.txt',
        'pretrained_weight_path': r'/home/weimf/cornernet/script/net_cls_1.pth',
        'gpu_ids':[0],
        
        }