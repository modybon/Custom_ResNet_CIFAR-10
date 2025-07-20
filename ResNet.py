#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# In[3]:


class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.stride = stride
        
        self.downsample = None
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            identity = self.downsample(identity)
        #print(f'Identity: {identity.shape}, Out: {out.shape}')
        out += identity
        out = self.relu(out)
        return out


# In[4]:


sample = torch.randn(2,3, 64, 64)

res_block = Residual_Block(3,128, stride=2)
y = res_block(sample)

print(y.shape)


# In[7]:


class Cifar_Resnet(nn.Module):
    def __init__(self,in_channels,classes_num):
        super(Cifar_Resnet, self).__init__()
        
        self.out_channels = 64
        self.conv_1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        # Residual layers
        self.layer1 = self._make_layer(self.out_channels, self.out_channels, 2, stride=1)
        self.layer2 = self._make_layer(self.out_channels, self.out_channels*2, 2, stride=2)
        self.layer3 = self._make_layer(self.out_channels, self.out_channels*2, 2, stride=2)
        self.layer4 = self._make_layer(self.out_channels, self.out_channels*2, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.out_channels, classes_num)
    def _make_layer(self, in_c, out_c, num_blocks, stride=1):
        layers = []
        layers.append(Residual_Block(in_c, out_c, stride=stride))
        for _ in range(1,num_blocks):
            layers.append(Residual_Block(out_c, out_c, stride=1))
        self.out_channels = out_c
        return nn.Sequential(*layers)
        
    def forward(self,x):
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# In[ ]:




