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

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0) 
        # outchanel  = in /8 , kSize(1,1) nghĩa là tích chập 1 ô chiều dài lẫn rộng , stride = 1 mỗi lần di chuyển 1 pixel từ chỗ đã xử lý padding ko thêm
        """_summary_
        """        # in_channels /8 là để chia nhỏ input channels để phân giải nhiều hơn
        # tại sao lại phải chia outchannel làm 8
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))# torch,zeros(1) có nghĩa là tạo mảng 1 chiều không có dữ liệu
        self.sigmoid = nn.Sigmoid() # dẫn sigmoid
        
    def forward(self, x):
        batch_size, channel, height, width = x.size() # show down dữ liệu trong mảng ra 

        axis = 1 # luu lại theo model truyền vào
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis) 

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)# hàm permute sắp xếp kết quả đâ có theo thứ tự của giá trị cũ thành 0 2 1 
        # biến 0 2 1 là batch size  image height, image width
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key) # hàm gộp 2 kqua lại để cho ra kết quả tốt hơn
        attention = self.sigmoid(attention_map) # biến giá trị của attention map thành số xác xuất để tìm ra kq tốt
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1)) # kết quả xuất ra 
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x # * với out vì out đc set là 1 mảng 0  , + x để lấy ra đc height width , channel
        return out # out ra array