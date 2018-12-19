# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:40:27 2018

@author: 60236
"""
import torch
from module.models import corner_net
model = corner_net(10)
x = torch.randn(1,3,128,128)
y = model(x)


from module.loss_module import mul_task_loss
import torchvision.transforms as transforms
from datasets.datasets import ListDataset

root = r'E:\遥感数据集\NWPU_VHR-10_dataset\yolo_label.txt'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])

dataset = ListDataset(root,img_size=128, fmp_size=32, transform=transform, classes=10)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=dataset.collate_fn)

debug = False

for idx, a in enumerate(dataloader):
    
    x = a['inputs']
    y_ = a['targets']
    break
loss = mul_task_loss()
#l = loss(y,y_)
#l[0].backward()

#from module.corner_pooling import left_pool
#import torch
#import torch.nn as nn
#
#class pool(nn.Module):
#    def __init__(self):
#        super(pool,self).__init__()
#        self.conv = nn.Conv2d(3,64,1)
#        self.pool = left_pool()
#        
#    def forward(self,x):
#        x = self.conv(x)
#        x = self.pool(x)
#        return x
#    
#    
#x = torch.randn(1,3,64,64)
#n = pool()
#y = n(x)
#
#loss = nn.L1Loss()
#t = torch.randn(1,64,64,64)
#
#
#l = loss(y,t)
#l.backward()



        
    
    
    
    
    
    






