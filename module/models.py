# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:49:24 2018

@author: 60236
"""
import torch.nn as nn
from .layers import Residual, conv_bn, conv_bn_relu, conv_relu
from .corner_pooling import top_pool, left_pool,bottom_pool, right_pool

class Hourglass(nn.Module): ###不改变特征图尺寸与深度
    def __init__(self, depth, nFeat, num_res=5, resBlock=Residual):
        super(Hourglass,self).__init__()
        self.depth = depth
        self.nFeat = nFeat
        self.num_res = num_res  # num residual modules per location
        self.resBlock = resBlock

        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])
    
    def _make_hour_glass(self):
        hg = []
        for i in range(self.depth):
            res = [self._make_residual(self.num_res) for _ in range(3)]  # skip(upper branch); down_path, up_path(lower branch)
            if i == (self.depth - 1):
                res.append(self._make_residual(self.num_res))  # extra one for the middle
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)
    
    def _hour_glass_forward(self, depth_id, x):
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == (self.depth - 1):
            low2 = self.hg[depth_id][3](low1)
        else:
            low2 = self._hour_glass_forward(depth_id + 1, low1)
        low3 = self.hg[depth_id][2](low2)
        up2 = self.upsample(low3)
        return up1 + up2
    def forward(self,x):
        return self._hour_glass_forward(0,x)

class HourglassNet(nn.Module):
    def __init__(self,nFeat=256, out_planes=256, num_res=1, inplanes=3, resBlock=Residual):
        super(HourglassNet,self).__init__()

        self.num_res = num_res
        self.nFeat = nFeat
        self.out_planes = out_planes
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.make_head()
    
        self.out = []
        self.out.append(Hourglass(4, nFeat, num_res, resBlock))
        self.out.append(self._make_residual(num_res))
        self.out.append(self._make_fc(nFeat, nFeat))
        self.out.append(nn.Conv2d(nFeat, out_planes, 1))
        self.out = nn.Sequential(*self.out)

    def make_head(self):
        self.conv1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = self.resBlock(128, 128)
        self.res3 = self.resBlock(128, self.nFeat)
    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])

    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True))

    def forward(self, x):
        # head
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)# ?,64,128,128
        
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)# ?,128,64,64
        x = self.res3(x)
        
        x = self.out(x)
        return x
        
class corner_net(nn.Module):
    def __init__(self, num_classes, inplanes=256, backbone=HourglassNet,num_res=7):
        super(corner_net,self).__init__()
        self.features = backbone(nFeat=256,out_planes=inplanes,num_res=num_res)
        self.relu = nn.ReLU(inplace=True)
        self.num_classes = num_classes    

        self.conv_bn_relu = conv_bn_relu(inplanes,inplanes)
        self.conv_bn = conv_bn(inplanes,inplanes)
        self.conv_bn_1x1 = conv_bn(inplanes,inplanes,1,1,0)
        self.conv_relu = conv_relu(inplanes,inplanes)
        self.out_ht = nn.Conv2d(inplanes, num_classes,1,1,0)
        self.out_eb = nn.Conv2d(inplanes,1,1,1,0)
        self.out_of = nn.Conv2d(inplanes,2,1,1,0)
        
    def forward(self,x):
        x = self.features(x)
        
        ##top-left
        a = self.conv_bn_relu(x)
        a = top_pool()(a)
        b = self.conv_bn_relu(x)
        b = left_pool()(b)
        ab = self.conv_bn(a+b)
        c = self.conv_bn_1x1(x)
        out = self.conv_bn_relu(self.relu(c+ab))
        
        heatmaps_tl   = self.out_ht(self.conv_relu(out))
        embeddings_tl = self.out_eb(self.conv_relu(out))
        offsets_tl    = self.out_of(self.conv_relu(out))
        
        ##bottem-right
        i = self.conv_bn_relu(x)
        i = bottom_pool()(i) ## cornerpooling
        j = self.conv_bn_relu(x)
        j = right_pool()(j) ## cornerpooling
        ij = self.conv_bn(i+j)
        k = self.conv_bn_1x1(x)
        out = self.conv_bn_relu(self.relu(k+ij))
        
        heatmaps_br   = self.out_ht(self.conv_relu(out))
        embeddings_br = self.out_eb(self.conv_relu(out))
        offsets_br    = self.out_of(self.conv_relu(out))

        return [heatmaps_tl,heatmaps_br,embeddings_tl,embeddings_br,offsets_tl,offsets_br]
  