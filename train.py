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
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print(args)

if len(config['gpu_ids']) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_ids'][0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.resume:
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    best_loss = float('inf')
    start_epoch = 0


# Data
print('==> Preparing data...')
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
                            ])
    
dataset = ListDataset(config['train_root'],img_size=config['image_size'], fmp_size=config['fms_size'], 
                       classes=config['num_classes'], train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataset.collate_fn)
net = network(config, lr=args.lr, resume=args.resume, device=device)

def train(epoch):
    train_loss = 0.
    for idx, ip_dict in enumerate(trainloader):
        images = ip_dict['inputs'].to(device)
        targets = [t.to(device) for t in ip_dict['targets']]
        
        loss, log_loss = net(images,targets)
        
        train_loss += loss.item()
        print('[Epoch %d, Batch %d/%d] [totalLoss:%.6f] [ht_loss:%.6f, off_loss:%.6f, pull_loss:%.6f, push_loss:%.6f]'
              %(epoch, idx, len(trainloader), loss.item(), log_loss[0], log_loss[1], log_loss[2], log_loss[3]))
    
    global best_loss
    train_loss /= len(trainloader)
    if train_loss < best_loss:
        print('saving...')
        if len(config['gpu_ids'])>1:
            w = net.model.module.state_dict()
        else:
            w = net.model.state_dict()
        state = {
                'weights': w,
                'loss': train_loss,
                'epoch': epoch,
                }
        os.makedirs(config['save_dir'], exist_ok=True)
        torch.save(state,config['save_dir'])
        best_loss = train_loss
    
def test(epoch):
    pass


#if __name__=="__main__":
#    for epoch in range(start_epoch, start_epoch+200):
#        train(epoch)
#        test(epoch)
#
#    pass







