# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:17:13 2021

@author: angelou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.conv_layer import Conv
from lib.self_attention import self_attn
import math

# dầu tiên chạy tích chập x1 để làm cho hình ảnh 
class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__() # ham init 
        self.conv0 = Conv(in_channel, out_channel, kSize=1,stride=1,padding=0) # kernel_size =1 => 1 lớp cho input channels đầu vào 
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3),stride = 1, padding=1) # tích chập 3x3 => input 3 lớp RGB 
        self.Hattn = self_attn(out_channel, mode='h') # chạy ham attention lấy ra out channel thei dạng height
        self.Wattn = self_attn(out_channel, mode='w') 

    def forward(self, x): # x là input model (node đâu vào , node out)
        x = self.conv0(x) 
        x = self.conv1(x)

        Hx = self.Hattn(x) # output về 1 mảng đc nén gồm batch size w h channel
        Wx = self.Wattn(Hx)#  

        return Wx
    
    # conv0 ( 128, 128, 2) -> activation relu.