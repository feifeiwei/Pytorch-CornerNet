# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:51:07 2018

@author: 60236
"""
import os
import torch
import argparse

from config import config
from Network import network
from datasets.datasets import ListDataset

import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch CornerNet Training')
parser.add_argument('--lr', default=3e-5, type=float, help='learning rate') 
parser.add_argument('--batch_size', type=int, default=2, help='size of each image batch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print(args)

if len(config['gpu_ids']) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_ids'][0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data
print('==> Preparing data...')
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
                            ])
     
dataset = ListDataset(config['root'],config['train_root'],img_size=config['image_size'], fmp_size=config['fms_size'], 
                       classes=config['num_classes'], train=True,transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataset.collate_fn)
net = network(config, lr=args.lr, resume=args.resume, device=device, train_loader=train_loader)



if __name__=="__main__":
    net.train()






