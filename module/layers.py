# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:55:13 2018

@author: 60236
"""

import torch.nn as nn


class Residual(nn.Module):
    def __init__(self,ins,outs):
        super(Residual,self).__init__()
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins,outs//2,1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2,outs//2,3,1,1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2,outs,1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins,outs,1)
        self.ins = ins
        self.outs = outs
    def forward(self,x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x

class conv_bn(nn.Module):
    def __init__(self,inplanes,outplanes,kernel=3,stride=1,pad=1):
        super(conv_bn,self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(inplanes,outplanes,kernel_size=kernel,stride=stride,padding=pad),
            nn.BatchNorm2d(outplanes)
        )
    def forward(self,x):
        return self.convBlock(x)

    
class conv_bn_relu(nn.Module):
    def __init__(self, inplanes,outplanes):
        super(conv_bn_relu,self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(inplanes,outplanes,3,1,1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.convBlock(x)
    
class conv_relu(nn.Module):
    def __init__(self, inplanes,outplanes):
        super(conv_relu,self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(inplanes,outplanes,3,1,1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.convBlock(x)
    
        
        