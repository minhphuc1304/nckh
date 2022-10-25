# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:15:44 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.conv_layer import Conv
import math

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0) # outchanel  = in /8
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid() # dẫn sigmoid
        
    def forward(self, x):
        batch_size, channel, height, width = x.size() # batch size la 128 

        axis = 1 # đặt biến để dễ dàng lấy dữ liệu // mõi lần chạy lại là 1 biến mới ko trùng lặp
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis) # channnel  = -1 ảnh hoán đổi màu sắc từ trắng thành đen

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)# hàm permute sắp xếp kết quả đâ có theo thứ tự của giá trị cũ thành 0 2 1
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key) # hàm gộp 2 kqua lại để cho ra kết quả tốt hơn
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1)) # kết quả xuất ra 
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x # * với out vì out đc set là 1 mảng 0  , + x để lấy ra đc height width , channel
        return out