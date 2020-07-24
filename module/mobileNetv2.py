import pdb
import torch.nn.functional as F
import torch
import torch.nn as nn
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        #assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        
        # make it nn.Sequential
        self.features_head = nn.Sequential(*self.features[:4])
        self.features0 = nn.Sequential(*self.features[4:7])
        self.features1 = nn.Sequential(*self.features[7:14])
        self.features2 = nn.Sequential(*self.features[14:19])

        #self._initialize_weights()
        self.toplayer = nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        
        # Lateral layers
        self.latlayer1 = nn.Conv2d(96, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 32, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 24, 256, kernel_size=1, stride=1, padding=0)
        
        self.smooth = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x, target=None, target_lanes=None, target_h=None):
        # bottom up
        c2 = self.features_head(x)
        c3 = self.features0(c2)  #torch.Size([1, 32, 28, 28]) /8
        c4 = self.features1(c3) #torch.Size([1, 96, 14, 14]) / 16
        c5 = self.features2(c4) #torch.Size([1, 1280, 7, 7]) /32
        
        # top down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        
        p2 = self.smooth(p2)
        
        return p2
        

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        #return F.upsample(x, size=(H,W), mode='bilinear') + y
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
if __name__=="__main__":
    x = torch.randn(1,3,256,256)
    n = MobileNetV2()
    
    y = n(x)
    print(y.shape)
    
    #torch.save(n.state_dict(), "aa.pth")
    
    