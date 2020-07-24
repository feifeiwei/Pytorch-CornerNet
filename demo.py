# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:55:09 2020

@author: 60236
"""

import os
import torch
import argparse

from config import config
from Network import network


parser = argparse.ArgumentParser(description='PyTorch CornerNet Demo')
parser.add_argument('--conf_thres', default=0.01, type=float, help='object threshold')
parser.add_argument('--nms_thres', type=float, default=0.3, help='iou threshold')
parser.add_argument('--test_path', type=str, default=r'E:\datasets\test', help='resume from checkpoint')
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = network(config, lr=None, resume=True, device=device, train_loader=None,mode='demo')

net.simple_demo(imgs_dir=args.test_path, conf_thres=args.conf_thres, nms_thres=args.nms_thres, num_dets=1000,ae_threshold=0.1)
