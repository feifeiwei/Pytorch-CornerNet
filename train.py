# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:51:07 2018

@author: 60236
"""
import os
import argparse
import torch
import torchvision.transforms as transforms
from Network import network
from datasets.datasets import ListDataset


parser = argparse.ArgumentParser(description='PyTorch cornernet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--image_shape', type=int, default=511, help='shape of each input image')
parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print(args)

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
root = r'E:\datasets\NWPU_VHR-10_dataset\train.txt'

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
                            ])
dataset = ListDataset(root,img_size=args.image_shape, fmp_size=128, classes=10, train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,collate_fn=dataset.collate_fn)

net = network(pull_weight=0.1, push_weight=0.1, offset_weight=1, lr=args.lr, resume=args.resume, device=device)

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
        state = {
                'weights': net.state_dict(),
                'loss': train_loss,
                'epoch': epoch,
                }
        os.makedirs('checkpoint', exist_ok=True)
        torch.save(state,'./checkpoint/ckpt.pth')
        best_loss = train_loss
    
def test(epoch):
    pass


if __name__=="__main__":
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)










