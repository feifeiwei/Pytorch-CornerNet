# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:49:24 2018

@author: 60236
"""
import torch.nn as nn

from .backbone import ResNet50
from .layers import  conv_bn, conv_bn_relu, conv_relu
from .corner_pooling import top_pool, left_pool,bottom_pool, right_pool

        
class corner_net(nn.Module):
    def __init__(self, num_classes, inplanes=256, backbone=ResNet50):
        super(corner_net,self).__init__()
        self.features = backbone()
        self.relu = nn.ReLU(inplace=True)
        self.num_classes = num_classes    

        self.conv_bn_relu1 = conv_bn_relu(inplanes,inplanes)
        self.conv_bn_relu2 = conv_bn_relu(inplanes,inplanes)
        self.conv_bn_relu3 = conv_bn_relu(inplanes,inplanes)
        
        self.conv_bn_relu4 = conv_bn_relu(inplanes,inplanes)
        self.conv_bn_relu5 = conv_bn_relu(inplanes,inplanes)
        self.conv_bn_relu6 = conv_bn_relu(inplanes,inplanes)
        
        self.conv_bn_tl = conv_bn(inplanes,inplanes)
        self.conv_bn_br = conv_bn(inplanes,inplanes)
        
        self.conv_bn_1x1_tl = conv_bn(inplanes,inplanes,1,1,0)
        self.conv_bn_1x1_br = conv_bn(inplanes,inplanes,1,1,0)
        
        self.conv_relu1 = conv_relu(inplanes,inplanes)
        self.conv_relu2 = conv_relu(inplanes,inplanes)
        self.conv_relu3 = conv_relu(inplanes,inplanes)
        self.conv_relu4 = conv_relu(inplanes,inplanes)
        self.conv_relu5 = conv_relu(inplanes,inplanes)
        self.conv_relu6 = conv_relu(inplanes,inplanes)
        
        self.out_ht_tl = nn.Conv2d(inplanes, num_classes,1,1,0)
        self.out_ht_br = nn.Conv2d(inplanes, num_classes,1,1,0)
        
        self.out_eb_tl = nn.Conv2d(inplanes,1,1,1,0)
        self.out_eb_br = nn.Conv2d(inplanes,1,1,1,0)
        
        self.out_of_tl = nn.Conv2d(inplanes,2,1,1,0)
        self.out_of_br = nn.Conv2d(inplanes,2,1,1,0)
        
    def forward(self,x):
        x = self.features(x)
        
        ##top-left
        a = self.conv_bn_relu1(x)
        a = top_pool()(a)
        b = self.conv_bn_relu2(x)
        b = left_pool()(b)
        ab = self.conv_bn_tl(a+b)
        c = self.conv_bn_1x1_tl(x)
        out = self.conv_bn_relu3(self.relu(c+ab))
        
        heatmaps_tl   = self.out_ht_tl(self.conv_relu1(out))
        embeddings_tl = self.out_eb_tl(self.conv_relu2(out))
        offsets_tl    = self.out_of_tl(self.conv_relu3(out))
        
        ##bottem-right
        i = self.conv_bn_relu4(x)
        i = bottom_pool()(i) 
        j = self.conv_bn_relu5(x)
        j = right_pool()(j)
        ij = self.conv_bn_br(i+j)
        k = self.conv_bn_1x1_br(x)
        out = self.conv_bn_relu6(self.relu(k+ij))
        
        heatmaps_br   = self.out_ht_br(self.conv_relu4(out))
        embeddings_br = self.out_eb_br(self.conv_relu5(out))
        offsets_br    = self.out_of_br(self.conv_relu6(out))

        return [heatmaps_tl,heatmaps_br,embeddings_tl,embeddings_br,offsets_tl,offsets_br]

    