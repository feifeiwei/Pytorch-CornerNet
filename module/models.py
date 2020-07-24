# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:49:24 2018

@author: 60236
"""
import torch.nn as nn
from .corner_pooling import *
from .backbone import ResNet50,ResNet18
from .mobileNetv2 import MobileNetV2
from .layers import  conv_bn, conv_bn_relu, conv_relu
#from corner_pooling import top_pool, left_pool,bottom_pool, right_pool

        
class corner_net(nn.Module):
    def __init__(self, num_classes, inplanes=256, backbone=MobileNetV2):
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
        #a = top_pool()(a)
        b = self.conv_bn_relu2(x)
       # b = left_pool()(b)
        ab = self.conv_bn_tl(a+b)
        c = self.conv_bn_1x1_tl(x)
        out = self.conv_bn_relu3(self.relu(c+ab))
        
        heatmaps_tl   = self.out_ht_tl(self.conv_relu1(out))
        embeddings_tl = self.out_eb_tl(self.conv_relu2(out))
        offsets_tl    = self.out_of_tl(self.conv_relu3(out))
        
        ##bottem-right
        i = self.conv_bn_relu4(x)
        #i = bottom_pool()(i) 
        j = self.conv_bn_relu5(x)
        #j = right_i.shapepool()(j)
        ij = self.conv_bn_br(i+j)
        k = self.conv_bn_1x1_br(x)
        out = self.conv_bn_relu6(self.relu(k+ij))
        
        heatmaps_br   = self.out_ht_br(self.conv_relu4(out))
        embeddings_br = self.out_eb_br(self.conv_relu5(out))
        offsets_br    = self.out_of_br(self.conv_relu6(out))

        return [heatmaps_tl,heatmaps_br,embeddings_tl,embeddings_br,offsets_tl,offsets_br]
    
    
if __name__=="__main__":
    import torch
    x = torch.randn(1,3,256,256)
    net = corner_net(num_classes=2)
    y = net(x)
    for i in y:
        print(i.shape)
   
    
    
    import torch.nn.init as init
    ####################################################################
    #n = torch.randn(1,3,224,224)
    
    
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    
    
    
    dd = net.state_dict()
    d = torch.load("../mobilenetv2_1.0-f2a8633.pth.tar",map_location=torch.device('cpu'))
    idx = 0
    
    for k in d.keys():
        dd[idx] = k
        print(dd[idx],"|", (k))
        idx += 1
    
    torch.save(net.state_dict(), 'pre_trained.pth')
    print("done")
    
    
    
    

    