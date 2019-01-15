# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:56:17 2018

@author: 60236
"""
import torch
import torch.optim as optim
from module.models import corner_net
from module.utils import weights_init_normal, detect
from module.loss_module import mul_task_loss

class network(torch.nn.Module):
    
    def __init__(self, cfg, lr=0.001, resume=False, device=None):
        super(network, self).__init__()

        self.classes = cfg['num_classes']
        
        self.detection = detect
        
        self.criterion = mul_task_loss(cfg['pull_weight'], cfg['push_weight'],cfg['offset_weight'])
        self.model = corner_net(self.classes)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        
        self.model.to(device)
        
        self.state_dict = self.model.state_dict()
        if resume:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.model.load_state_dict(checkpoint['net'])

        else:
            print("==> Random init model...")
            self.model.apply(weights_init_normal)
        
    
        
    def forward(self, x, target=None):
        if target:
            self.optimizer.zero_grad()
            self.model.train()
            output = self.model(x)
            loss, log = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            return loss, log
        
        else:
            self.model.eval()
            output = self.model(x)
            detections = self.detection(*output)
            return detections
        
        
        
        
        
    