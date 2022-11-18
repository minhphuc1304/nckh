# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:12:46 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):  # conv
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti: # nêu >0 chạy
            self.bn_relu = BNPReLU(nOut) # lọc các giá trị <0

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output # > 0 


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3) # chuẩn hóa giá trị nhất có thể
        self.acti = nn.PReLU(nIn) 

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

# https://aicurious.io/posts/2019-09-23-cac-ham-kich-hoat-activation-function-trong-neural-networks/
# batch_norm : https://www.phamduytung.com/blog/2022-02-25-normalization/
# https://d2l.aivivn.com/chapter_convolutional-modern/batch-norm_vn.html