# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:34:46 2018

@author: 60236
"""

config = {
        'image_size':416,
        'fms_size':208,
        
        'pull_weight':0.1, 
        'push_weight':0.1, 
        'offset_weight':1,    
        
        'num_classes':2,
        
        'save_dir':r'./checkpoint/ckpt.pth',
        
        'train_root': r'E:\遥感车辆数据集\UCAS\test.txt',
        'pretrained_weight_path':r'/home/weimf/cornernet/script/net_cls_1.pth',
        'gpu_ids':[0,1,2,3],
        }