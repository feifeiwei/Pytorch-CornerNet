# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:56:17 2018

@author: 60236
"""
import time
import datetime
import os
import pdb
import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from PIL import Image
from module.models import corner_net
from module.utils import  detect, non_max_suppression
from module.loss_module import mul_task_loss


class network(torch.nn.Module):
    
    def __init__(self,cfg, lr=None, resume=False, device=None, train_loader=None,mode='train'):
        super(network, self).__init__()

        self.num_classes = cfg['num_classes']
        self.classes_name = cfg['classes']
        self.gpu_ids = cfg['gpu_ids']
        self.detection = detect
        self.epochs = cfg['epochs']
        self.device = device
        self.save_dir = cfg['save_dir']
        
        self.model = corner_net(self.num_classes)
        
        if mode=='train':
            self.criterion = mul_task_loss(cfg['pull_weight'], cfg['push_weight'],cfg['offset_weight'])
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            self.train_loader = train_loader
        
            if len(self.gpu_ids)>1:
                self.model = nn.DataParallel(self.model,device_ids=self.gpu_ids)
                self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.gpu_ids)
                
            if resume:
                print('==> Resuming from checkpoint..')
                checkpoint = torch.load('./checkpoint/ckpt.pth')
                self.model.load_state_dict(checkpoint['weights'])
                self.start_epoch = checkpoint['start_epoch']
                self.best_loss = checkpoint['best_loss']
            else:
                print("==> Random init model...")
                #init = torch.load(r'../script/net_cls_2.pth')
                #self.model.load_state_dict(init)
                self.start_epoch = 0
                self.best_loss = float("inf")
        elif mode=='demo':
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.model.load_state_dict(checkpoint['weights'])
            print("checkpoint load successful..")
        self.model.to(device)
        
        
        
    
    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch+self.epochs):
            train_loss = 0.
            self.model.train()
            for idx, ip_dict in enumerate(self.train_loader):
                images = ip_dict['inputs'].to(self.device)
                targets = [t.to(self.device) for t in ip_dict['targets']]
                self.optimizer.zero_grad()
                output = self.model(images)
                #import pdb
                #pdb.set_trace()
                loss, log_loss = self.criterion(output, targets)
              
                
                train_loss += loss.item()
                print('[Epoch %d, Batch %d/%d] [totalLoss:%.6f] [ht_loss:%.6f, off_loss:%.6f, pull_loss:%.6f, push_loss:%.6f]'
                      %(epoch, idx, len(self.train_loader), loss.item(), log_loss[0], log_loss[1], log_loss[2], log_loss[3]))
            
            global best_loss
            train_loss /= len(self.train_loader)
            if train_loss < best_loss:
                print('saving...')
                w = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
                state = {
                        'weights': w,
                        'best_loss': train_loss,
                        'start_epoch': epoch,
                        }
                #os.makedirs(self.save_dir, exist_ok=True)
                torch.save(state, self.save_dir)
                best_loss = train_loss
        
        
    def simple_demo(self, imgs_dir, conf_thres=0.5, nms_thres=0.4,num_dets=1000,ae_threshold=0.5):
        
        files = glob.glob(os.path.join(imgs_dir,'*.png'))
        print("\nPerforming object detection..")
        pre_time = time.time()
        
        for ind, img_dir in enumerate(files):
            img = cv2.imread(img_dir)
            img_array = self.get_img_np_nchw(img_dir)
            inputs = torch.from_numpy(img_array).float().cuda()
            output = self.model(inputs)
            detections = self.detection(*output, k=100, ae_threshold=ae_threshold, num_dets=num_dets, down_ratio=4)
            detections = non_max_suppression(detections,num_classes=self.num_classes, conf_thres=conf_thres, nms_thres=nms_thres)
  
            detections = detections[0]
            
            if detections == None:
                continue
            for x1,y1,x2,y2, conf, l_conf, r_conf, cls in detections:
                
                print("\t+ Label: %s, conf: %.3f"%(self.classes_name[int(cls)], conf))
                color = [0,255,0]
                cv2.rectangle(img, (x1,y1), (x2,y2), color,2)
            
            cur_time = time.time()
            Inference_time = datetime.timedelta(seconds=cur_time-pre_time)
            print('\t+ Batch %d, Inference Time: %s' %(ind, Inference_time))
            print("saving img： %d"%ind)
            cv2.imwrite("output/%s.png"%ind, img)
            pre_time = cur_time
            
    def get_img_np_nchw(self,filename):
 
        image = Image.open(filename).convert('RGB').resize((416, 416))
        miu = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = np.array(image, dtype=float) / 255.
        r = (img_np[:,:,0]- miu[0]) / std[0]
        g = (img_np[:,:,1]- miu[1]) / std[1]
        b = (img_np[:,:,2]- miu[2]) / std[2]
        img_np_t = np.array([r,g,b])#.transpose(1,2,0)
        
        img_np_nchw = np.expand_dims(img_np_t, axis=0)

        return img_np_nchw
        
        
        
        
        
        
    