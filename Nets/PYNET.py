# -*- coding:utf8 -*-
# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict


class PyNET(nn.Module):

    def __init__(self, level, channel=32, instance_norm=True, instance_norm_level_1=False):
        super(PyNET, self).__init__()

        self.level = level

        self.conv_l1_d1 = ConvMultiBlock(4, channel, 3, instance_norm=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv_l2_d1 = ConvMultiBlock(channel, channel * 2, 3, instance_norm=instance_norm)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv_l3_d1 = ConvMultiBlock(channel * 2, channel * 4, 3, instance_norm=instance_norm)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv_l4_d1 = ConvMultiBlock(channel * 4, channel * 8, 3, instance_norm=instance_norm)
        self.pool4 = nn.MaxPool2d(2, 2)

        # -------------------------------------

        self.conv_l5_d1 = ConvMultiBlock(channel * 8, channel * 16, 3, instance_norm=instance_norm)
        self.conv_l5_d2 = ConvMultiBlock(channel * 16, channel * 16, 3, instance_norm=instance_norm)
        self.conv_l5_d3 = ConvMultiBlock(channel * 16, channel * 16, 3, instance_norm=instance_norm)
        self.conv_l5_d4 = ConvMultiBlock(channel * 16, channel * 16, 3, instance_norm=instance_norm)

        self.conv_t4a = UpsampleConvLayer(channel * 16, channel * 8, 3)
        self.conv_t4b = UpsampleConvLayer(channel * 16, channel * 8, 3)

        self.conv_l5_out = ConvLayer(channel * 16, 3, kernel_size=3, stride=1, relu=False)
        self.output_l5 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l4_d3 = ConvMultiBlock(channel * 16, channel * 8, 3, instance_norm=instance_norm)
        self.conv_l4_d4 = ConvMultiBlock(channel * 8, channel * 8, 3, instance_norm=instance_norm)
        self.conv_l4_d5 = ConvMultiBlock(channel * 8, channel * 8, 3, instance_norm=instance_norm)
        self.conv_l4_d6 = ConvMultiBlock(channel * 8, channel * 8, 3, instance_norm=instance_norm)
        self.conv_l4_d8 = ConvMultiBlock(channel * 16, channel * 8, 3, instance_norm=instance_norm)

        self.conv_t3a = UpsampleConvLayer(channel * 8, channel * 4, 3)
        self.conv_t3b = UpsampleConvLayer(channel * 8, channel * 4, 3)

        self.conv_l4_out = ConvLayer(channel * 8, 3, kernel_size=3, stride=1, relu=False)
        self.output_l4 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l3_d3 = ConvMultiBlock(channel * 8, channel * 4, 5, instance_norm=instance_norm)
        self.conv_l3_d4 = ConvMultiBlock(channel * 8, channel * 4, 5, instance_norm=instance_norm)
        self.conv_l3_d5 = ConvMultiBlock(channel * 8, channel * 4, 5, instance_norm=instance_norm)
        self.conv_l3_d6 = ConvMultiBlock(channel * 8, channel * 4, 5, instance_norm=instance_norm)
        self.conv_l3_d8 = ConvMultiBlock(channel * 16, channel * 4, 3, instance_norm=instance_norm)

        self.conv_t2a = UpsampleConvLayer(channel * 4, channel * 2, 3)
        self.conv_t2b = UpsampleConvLayer(channel * 4, channel * 2, 3)

        self.conv_l3_out = ConvLayer(channel * 4, 3, kernel_size=3, stride=1, relu=False)
        self.output_l3 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l2_d3 = ConvMultiBlock(channel * 4, channel * 2, 5, instance_norm=instance_norm)
        self.conv_l2_d5 = ConvMultiBlock(channel * 6, channel * 2, 7, instance_norm=instance_norm)
        self.conv_l2_d6 = ConvMultiBlock(channel * 6, channel * 2, 7, instance_norm=instance_norm)
        self.conv_l2_d7 = ConvMultiBlock(channel * 6, channel * 2, 7, instance_norm=instance_norm)
        self.conv_l2_d8 = ConvMultiBlock(channel * 6, channel * 2, 7, instance_norm=instance_norm)
        self.conv_l2_d10 = ConvMultiBlock(channel * 8, channel * 2, 5, instance_norm=instance_norm)
        self.conv_l2_d12 = ConvMultiBlock(channel * 6, channel * 2, 3, instance_norm=instance_norm)

        self.conv_t1a = UpsampleConvLayer(channel * 2, channel, 3)
        self.conv_t1b = UpsampleConvLayer(channel * 2, channel, 3)

        self.conv_l2_out = ConvLayer(channel * 2, 3, kernel_size=3, stride=1, relu=False)
        self.output_l2 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l1_d3 = ConvMultiBlock(channel * 2, channel, 5, instance_norm=False)
        self.conv_l1_d5 = ConvMultiBlock(96, channel, 7, instance_norm=instance_norm_level_1)

        self.conv_l1_d6 = ConvMultiBlock(96, channel, 9, instance_norm=instance_norm_level_1)
        self.conv_l1_d7 = ConvMultiBlock(channel * 4, channel, 9, instance_norm=instance_norm_level_1)
        self.conv_l1_d8 = ConvMultiBlock(channel * 4, channel, 9, instance_norm=instance_norm_level_1)
        self.conv_l1_d9 = ConvMultiBlock(channel * 4, channel, 9, instance_norm=instance_norm_level_1)

        self.conv_l1_d10 = ConvMultiBlock(channel * 4, channel, 7, instance_norm=instance_norm_level_1)
        self.conv_l1_d12 = ConvMultiBlock(channel * 4, channel, 5, instance_norm=instance_norm_level_1)
        self.conv_l1_d14 = ConvMultiBlock(channel * 4, channel, 3, instance_norm=False)

        self.conv_l1_out = ConvLayer(channel, 3, kernel_size=3, stride=1, relu=False)
        self.output_l1 = nn.Sigmoid()

        self.conv_t0 = UpsampleConvLayer(channel, 16, 3)

        # -------------------------------------

        self.conv_l0_d1 = ConvLayer(16, 3, kernel_size=3, stride=1, relu=False)
        self.output_l0 = nn.Sigmoid()

    def level_5(self, pool4):

        conv_l5_d1 = self.conv_l5_d1(pool4)
        conv_l5_d2 = self.conv_l5_d2(conv_l5_d1)
        conv_l5_d3 = self.conv_l5_d3(conv_l5_d2)
        conv_l5_d4 = self.conv_l5_d4(conv_l5_d3)

        conv_t4a = self.conv_t4a(conv_l5_d4)
        conv_t4b = self.conv_t4b(conv_l5_d4)

        conv_l5_out = self.conv_l5_out(conv_l5_d4)
        output_l5 = self.output_l5(conv_l5_out)

        return output_l5, conv_t4a, conv_t4b

    def level_4(self, conv_l4_d1, conv_t4a, conv_t4b):

        conv_l4_d2 = torch.cat([conv_l4_d1, conv_t4a], 1)

        conv_l4_d3 = self.conv_l4_d3(conv_l4_d2)
        conv_l4_d4 = self.conv_l4_d4(conv_l4_d3) + conv_l4_d3
        conv_l4_d5 = self.conv_l4_d5(conv_l4_d4) + conv_l4_d4
        conv_l4_d6 = self.conv_l4_d6(conv_l4_d5)

        conv_l4_d7 = torch.cat([conv_l4_d6, conv_t4b], 1)
        conv_l4_d8 = self.conv_l4_d8(conv_l4_d7)

        conv_t3a = self.conv_t3a(conv_l4_d8)
        conv_t3b = self.conv_t3b(conv_l4_d8)

        conv_l4_out = self.conv_l4_out(conv_l4_d8)
        output_l4 = self.output_l4(conv_l4_out)

        return output_l4, conv_t3a, conv_t3b

    def level_3(self, conv_l3_d1, conv_t3a, conv_t3b):

        conv_l3_d2 = torch.cat([conv_l3_d1, conv_t3a], 1)

        conv_l3_d3 = self.conv_l3_d3(conv_l3_d2) + conv_l3_d2
        conv_l3_d4 = self.conv_l3_d4(conv_l3_d3) + conv_l3_d3
        conv_l3_d5 = self.conv_l3_d5(conv_l3_d4) + conv_l3_d4
        conv_l3_d6 = self.conv_l3_d6(conv_l3_d5)

        conv_l3_d7 = torch.cat([conv_l3_d6, conv_l3_d1, conv_t3b], 1)
        conv_l3_d8 = self.conv_l3_d8(conv_l3_d7)

        conv_t2a = self.conv_t2a(conv_l3_d8)
        conv_t2b = self.conv_t2b(conv_l3_d8)

        conv_l3_out = self.conv_l3_out(conv_l3_d8)
        output_l3 = self.output_l3(conv_l3_out)

        return output_l3, conv_t2a, conv_t2b

    def level_2(self, conv_l2_d1, conv_t2a, conv_t2b):

        conv_l2_d2 = torch.cat([conv_l2_d1, conv_t2a], 1)
        conv_l2_d3 = self.conv_l2_d3(conv_l2_d2)
        conv_l2_d4 = torch.cat([conv_l2_d3, conv_l2_d1], 1)

        conv_l2_d5 = self.conv_l2_d5(conv_l2_d4) + conv_l2_d4
        conv_l2_d6 = self.conv_l2_d6(conv_l2_d5) + conv_l2_d5
        conv_l2_d7 = self.conv_l2_d7(conv_l2_d6) + conv_l2_d6
        conv_l2_d8 = self.conv_l2_d8(conv_l2_d7)
        conv_l2_d9 = torch.cat([conv_l2_d8, conv_l2_d1], 1)

        conv_l2_d10 = self.conv_l2_d10(conv_l2_d9)
        conv_l2_d11 = torch.cat([conv_l2_d10, conv_t2b], 1)
        conv_l2_d12 = self.conv_l2_d12(conv_l2_d11)

        conv_t1a = self.conv_t1a(conv_l2_d12)
        conv_t1b = self.conv_t1b(conv_l2_d12)

        conv_l2_out = self.conv_l2_out(conv_l2_d12)
        output_l2 = self.output_l2(conv_l2_out)

        return output_l2, conv_t1a, conv_t1b

    def level_1(self, conv_l1_d1, conv_t1a, conv_t1b):

        conv_l1_d2 = torch.cat([conv_l1_d1, conv_t1a], 1)
        conv_l1_d3 = self.conv_l1_d3(conv_l1_d2)
        conv_l1_d4 = torch.cat([conv_l1_d3, conv_l1_d1], 1)

        conv_l1_d5 = self.conv_l1_d5(conv_l1_d4)

        conv_l1_d6 = self.conv_l1_d6(conv_l1_d5)
        conv_l1_d7 = self.conv_l1_d7(conv_l1_d6) + conv_l1_d6
        conv_l1_d8 = self.conv_l1_d8(conv_l1_d7) + conv_l1_d7
        conv_l1_d9 = self.conv_l1_d9(conv_l1_d8) + conv_l1_d8

        conv_l1_d10 = self.conv_l1_d10(conv_l1_d9)
        conv_l1_d11 = torch.cat([conv_l1_d10, conv_l1_d1], 1)

        conv_l1_d12 = self.conv_l1_d12(conv_l1_d11)
        conv_l1_d13 = torch.cat([conv_l1_d12, conv_t1b, conv_l1_d1], 1)

        conv_l1_d14 = self.conv_l1_d14(conv_l1_d13)
        conv_t0 = self.conv_t0(conv_l1_d14)

        conv_l1_out = self.conv_l1_out(conv_l1_d14)
        output_l1 = self.output_l1(conv_l1_out)

        return output_l1, conv_t0

    def level_0(self, conv_t0):

        conv_l0_d1 = self.conv_l0_d1(conv_t0)
        output_l0 = self.output_l0(conv_l0_d1)

        return output_l0

    def pack(self, img, R=0, Gr=0, Gb=0, B=0):
        kernel = torch.FloatTensor([[R, Gr], [Gb, B]])
        # kernel = kernel.unsqueeze(0).unsqueeze(0).cuda()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        weight = nn.Parameter(data=kernel, requires_grad=False)
        img = F.conv2d(img, weight, stride=2)

        return img

    def forward(self, x):

        R = self.pack(x, R=1)
        Gr = self.pack(x, Gr=1)
        Gb = self.pack(x, Gb=1)
        B = self.pack(x, B=1)
        x = torch.cat([B, Gb, R, Gr], dim=1)

        conv_l1_d1 = self.conv_l1_d1(x)
        pool1 = self.pool1(conv_l1_d1)

        conv_l2_d1 = self.conv_l2_d1(pool1)
        pool2 = self.pool2(conv_l2_d1)

        conv_l3_d1 = self.conv_l3_d1(pool2)
        pool3 = self.pool3(conv_l3_d1)

        conv_l4_d1 = self.conv_l4_d1(pool3)
        pool4 = self.pool4(conv_l4_d1)

        output_l5, conv_t4a, conv_t4b = self.level_5(pool4)

        if self.level < 5:
            output_l4, conv_t3a, conv_t3b = self.level_4(conv_l4_d1, conv_t4a, conv_t4b)
        if self.level < 4:
            output_l3, conv_t2a, conv_t2b = self.level_3(conv_l3_d1, conv_t3a, conv_t3b)
        if self.level < 3:
            output_l2, conv_t1a, conv_t1b = self.level_2(conv_l2_d1, conv_t2a, conv_t2b)
        if self.level < 2:
            output_l1, conv_t0 = self.level_1(conv_l1_d1, conv_t1a, conv_t1b)
        if self.level < 1:
            output_l0 = self.level_0(conv_t0)

        if self.level == 0:
            enhanced = output_l0
        if self.level == 1:
            enhanced = output_l1
        if self.level == 2:
            enhanced = output_l2
        if self.level == 3:
            enhanced = output_l3
        if self.level == 4:
            enhanced = output_l4
        if self.level == 5:
            enhanced = output_l5

        return enhanced

class PyNET_smaller(nn.Module):

    def __init__(self, level, channel=32, instance_norm=True, instance_norm_level_1=False):
        super(PyNET_smaller, self).__init__()

        self.level = level

        self.conv_l1_d1 = ConvMultiBlock(4, channel, 3, instance_norm=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv_l2_d1 = ConvMultiBlock(channel, channel * 2, 3, instance_norm=instance_norm)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv_l3_d1 = ConvMultiBlock(channel * 2, channel * 4, 3, instance_norm=instance_norm)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv_l4_d1 = ConvMultiBlock(channel * 4, channel * 8, 3, instance_norm=instance_norm)
        self.pool4 = nn.MaxPool2d(2, 2)

        # -------------------------------------

        self.conv_l5_d1 = ConvMultiBlock(channel * 8, channel * 16, 3, instance_norm=instance_norm)
        self.conv_l5_d4 = ConvMultiBlock(channel * 16, channel * 16, 3, instance_norm=instance_norm)

        self.conv_t4a = UpsampleConvLayer(channel * 16, channel * 8, 3)
        self.conv_t4b = UpsampleConvLayer(channel * 16, channel * 8, 3)

        self.conv_l5_out = ConvLayer(channel * 16, 3, kernel_size=3, stride=1, relu=False)
        self.output_l5 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l4_d3 = ConvMultiBlock(channel * 16, channel * 8, 3, instance_norm=instance_norm)
        self.conv_l4_d4 = ConvMultiBlock(channel * 8, channel * 8, 3, instance_norm=instance_norm)
        self.conv_l4_d6 = ConvMultiBlock(channel * 16, channel * 8, 3, instance_norm=instance_norm)

        self.conv_t3a = UpsampleConvLayer(channel * 8, channel * 4, 3)
        self.conv_t3b = UpsampleConvLayer(channel * 8, channel * 4, 3)

        self.conv_l4_out = ConvLayer(channel * 8, 3, kernel_size=3, stride=1, relu=False)
        self.output_l4 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l3_d3 = ConvMultiBlock(channel * 8, channel * 4, 5, instance_norm=instance_norm)
        self.conv_l3_d4 = ConvMultiBlock(channel * 8, channel * 4, 5, instance_norm=instance_norm)
        self.conv_l3_d6 = ConvMultiBlock(channel * 16, channel * 4, 3, instance_norm=instance_norm)

        self.conv_t2a = UpsampleConvLayer(channel * 4, channel * 2, 3)
        self.conv_t2b = UpsampleConvLayer(channel * 4, channel * 2, 3)

        self.conv_l3_out = ConvLayer(channel * 4, 3, kernel_size=3, stride=1, relu=False)
        self.output_l3 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l2_d3 = ConvMultiBlock(channel * 4, channel * 2, 5, instance_norm=instance_norm)
        self.conv_l2_d5 = ConvMultiBlock(channel * 6, channel * 2, 7, instance_norm=instance_norm)
        self.conv_l2_d6 = ConvMultiBlock(channel * 6, channel * 2, 7, instance_norm=instance_norm)
        self.conv_l2_d8 = ConvMultiBlock(channel * 8, channel * 2, 5, instance_norm=instance_norm)
        self.conv_l2_d10 = ConvMultiBlock(channel * 6, channel * 2, 3, instance_norm=instance_norm)

        self.conv_t1a = UpsampleConvLayer(channel * 2, channel, 3)
        self.conv_t1b = UpsampleConvLayer(channel * 2, channel, 3)

        self.conv_l2_out = ConvLayer(channel * 2, 3, kernel_size=3, stride=1, relu=False)
        self.output_l2 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l1_d3 = ConvMultiBlock(channel * 2, channel, 5, instance_norm=False)
        self.conv_l1_d5 = ConvMultiBlock(channel * 3, channel, 7, instance_norm=instance_norm_level_1)

        self.conv_l1_d6 = ConvMultiBlock(channel * 3, channel, 9, instance_norm=instance_norm_level_1)
        self.conv_l1_d7 = ConvMultiBlock(channel * 4, channel, 9, instance_norm=instance_norm_level_1)

        self.conv_l1_d8 = ConvMultiBlock(channel * 4, channel, 7, instance_norm=instance_norm_level_1)
        self.conv_l1_d10 = ConvMultiBlock(channel * 4, channel, 5, instance_norm=instance_norm_level_1)
        self.conv_l1_d12 = ConvMultiBlock(channel * 4, channel, 3, instance_norm=False)

        self.conv_l1_out = ConvLayer(channel, 3, kernel_size=3, stride=1, relu=False)
        self.output_l1 = nn.Sigmoid()

        self.conv_t0 = UpsampleConvLayer(channel, 16, 3)

        # -------------------------------------

        self.conv_l0_d1 = ConvLayer(16, 3, kernel_size=3, stride=1, relu=False)
        self.output_l0 = nn.Sigmoid()

    def level_5(self, pool4):

        conv_l5_d1 = self.conv_l5_d1(pool4)
        conv_l5_d2 = self.conv_l5_d4(conv_l5_d1)

        conv_t4a = self.conv_t4a(conv_l5_d2)
        conv_t4b = self.conv_t4b(conv_l5_d2)

        conv_l5_out = self.conv_l5_out(conv_l5_d2)
        output_l5 = self.output_l5(conv_l5_out)

        return output_l5, conv_t4a, conv_t4b

    def level_4(self, conv_l4_d1, conv_t4a, conv_t4b):

        conv_l4_d2 = torch.cat([conv_l4_d1, conv_t4a], 1)

        conv_l4_d3 = self.conv_l4_d3(conv_l4_d2)
        conv_l4_d4 = self.conv_l4_d4(conv_l4_d3)

        conv_l4_d5 = torch.cat([conv_l4_d4, conv_t4b], 1)
        conv_l4_d6 = self.conv_l4_d6(conv_l4_d5)

        conv_t3a = self.conv_t3a(conv_l4_d6)
        conv_t3b = self.conv_t3b(conv_l4_d6)

        conv_l4_out = self.conv_l4_out(conv_l4_d6)
        output_l4 = self.output_l4(conv_l4_out)

        return output_l4, conv_t3a, conv_t3b

    def level_3(self, conv_l3_d1, conv_t3a, conv_t3b):

        conv_l3_d2 = torch.cat([conv_l3_d1, conv_t3a], 1)

        conv_l3_d3 = self.conv_l3_d3(conv_l3_d2) + conv_l3_d2
        conv_l3_d4 = self.conv_l3_d4(conv_l3_d3)

        conv_l3_d5 = torch.cat([conv_l3_d4, conv_l3_d1, conv_t3b], 1)
        conv_l3_d6 = self.conv_l3_d6(conv_l3_d5)

        conv_t2a = self.conv_t2a(conv_l3_d6)
        conv_t2b = self.conv_t2b(conv_l3_d6)

        conv_l3_out = self.conv_l3_out(conv_l3_d6)
        output_l3 = self.output_l3(conv_l3_out)

        return output_l3, conv_t2a, conv_t2b

    def level_2(self, conv_l2_d1, conv_t2a, conv_t2b):

        conv_l2_d2 = torch.cat([conv_l2_d1, conv_t2a], 1)
        conv_l2_d3 = self.conv_l2_d3(conv_l2_d2)
        conv_l2_d4 = torch.cat([conv_l2_d3, conv_l2_d1], 1)

        conv_l2_d5 = self.conv_l2_d5(conv_l2_d4) + conv_l2_d4
        conv_l2_d6 = self.conv_l2_d6(conv_l2_d5)
        conv_l2_d7 = torch.cat([conv_l2_d6, conv_l2_d1], 1)

        conv_l2_d8 = self.conv_l2_d8(conv_l2_d7)
        conv_l2_d9 = torch.cat([conv_l2_d8, conv_t2b], 1)
        conv_l2_d10 = self.conv_l2_d10(conv_l2_d9)

        conv_t1a = self.conv_t1a(conv_l2_d10)
        conv_t1b = self.conv_t1b(conv_l2_d10)

        conv_l2_out = self.conv_l2_out(conv_l2_d10)
        output_l2 = self.output_l2(conv_l2_out)

        return output_l2, conv_t1a, conv_t1b

    def level_1(self, conv_l1_d1, conv_t1a, conv_t1b):

        conv_l1_d2 = torch.cat([conv_l1_d1, conv_t1a], 1)
        conv_l1_d3 = self.conv_l1_d3(conv_l1_d2)
        conv_l1_d4 = torch.cat([conv_l1_d3, conv_l1_d1], 1)

        conv_l1_d5 = self.conv_l1_d5(conv_l1_d4)

        conv_l1_d6 = self.conv_l1_d6(conv_l1_d5)
        conv_l1_d7 = self.conv_l1_d7(conv_l1_d6) + conv_l1_d6

        conv_l1_d8 = self.conv_l1_d8(conv_l1_d7)
        conv_l1_d9 = torch.cat([conv_l1_d8, conv_l1_d1], 1)

        conv_l1_d10 = self.conv_l1_d10(conv_l1_d9)
        conv_l1_d11 = torch.cat([conv_l1_d10, conv_t1b, conv_l1_d1], 1)

        conv_l1_d12 = self.conv_l1_d12(conv_l1_d11)
        conv_t0 = self.conv_t0(conv_l1_d12)

        conv_l1_out = self.conv_l1_out(conv_l1_d12)
        output_l1 = self.output_l1(conv_l1_out)

        return output_l1, conv_t0

    def level_0(self, conv_t0):

        conv_l0_d1 = self.conv_l0_d1(conv_t0)
        output_l0 = self.output_l0(conv_l0_d1)

        return output_l0

    def pack(self, img, R=0, Gr=0, Gb=0, B=0):
        kernel = torch.FloatTensor([[R, Gr], [Gb, B]])
        kernel = kernel.unsqueeze(0).unsqueeze(0).cuda()
        # kernel = kernel.unsqueeze(0).unsqueeze(0)
        weight = nn.Parameter(data=kernel, requires_grad=False)
        img = F.conv2d(img, weight, stride=2)

        return img

    def forward(self, x):

        global conv_t3a, conv_t3b, conv_t2a, conv_t2b, conv_t1a, conv_t1b, conv_t0, \
            output_l0, output_l1, output_l2, output_l3, output_l4, enhanced

        # R = self.pack(x, R=1)
        # Gr = self.pack(x, Gr=1)
        # Gb = self.pack(x, Gb=1)
        # B = self.pack(x, B=1)
        # x = torch.cat([R, Gr, Gb, B], dim=1)

        conv_l1_d1 = self.conv_l1_d1(x)
        pool1 = self.pool1(conv_l1_d1)

        conv_l2_d1 = self.conv_l2_d1(pool1)
        pool2 = self.pool2(conv_l2_d1)

        conv_l3_d1 = self.conv_l3_d1(pool2)
        pool3 = self.pool3(conv_l3_d1)

        conv_l4_d1 = self.conv_l4_d1(pool3)
        pool4 = self.pool4(conv_l4_d1)

        output_l5, conv_t4a, conv_t4b = self.level_5(pool4)

        if self.level < 5:
            output_l4, conv_t3a, conv_t3b = self.level_4(conv_l4_d1, conv_t4a, conv_t4b)
        if self.level < 4:
            output_l3, conv_t2a, conv_t2b = self.level_3(conv_l3_d1, conv_t3a, conv_t3b)
        if self.level < 3:
            output_l2, conv_t1a, conv_t1b = self.level_2(conv_l2_d1, conv_t2a, conv_t2b)
        if self.level < 2:
            output_l1, conv_t0 = self.level_1(conv_l1_d1, conv_t1a, conv_t1b)
        if self.level < 1:
            output_l0 = self.level_0(conv_t0)

        if self.level == 0:
            enhanced = output_l0
        if self.level == 1:
            enhanced = output_l1
        if self.level == 2:
            enhanced = output_l2
        if self.level == 3:
            enhanced = output_l3
        if self.level == 4:
            enhanced = output_l4
        if self.level == 5:
            enhanced = output_l5

        return enhanced


class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, max_conv_size, instance_norm):

        super(ConvMultiBlock, self).__init__()
        self.max_conv_size = max_conv_size

        self.conv_3a = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)
        self.conv_3b = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 5:
            self.conv_5a = ConvLayer(in_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)
            self.conv_5b = ConvLayer(out_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 7:
            self.conv_7a = ConvLayer(in_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)
            self.conv_7b = ConvLayer(out_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 9:
            self.conv_9a = ConvLayer(in_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)
            self.conv_9b = ConvLayer(out_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)

    def forward(self, x):

        out_3 = self.conv_3a(x)
        output_tensor = self.conv_3b(out_3)

        if self.max_conv_size >= 5:
            out_5 = self.conv_5a(x)
            out_5 = self.conv_5b(out_5)
            output_tensor = torch.cat([output_tensor, out_5], 1)

        if self.max_conv_size >= 7:
            out_7 = self.conv_7a(x)
            out_7 = self.conv_7b(out_7)
            output_tensor = torch.cat([output_tensor, out_7], 1)

        if self.max_conv_size >= 9:
            out_9 = self.conv_9a(x)
            out_9 = self.conv_9b(out_9)
            output_tensor = torch.cat([output_tensor, out_9], 1)

        return output_tensor


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, dilation=reflection_padding, stride=stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=2, stride=1, relu=True):

        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample)

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.upsample(x)
        out = self.reflection_pad(out)
        out = self.conv2d(out)

        if self.relu:
            out = self.relu(out)

        return out


# img = torch.rand((1, 1, 1056, 1920))
# model = PyNET_smaller(level=0, instance_norm=True, instance_norm_level_1=True)
# out = model(img)
# print(out.shape)

# from torchstat import stat
#
# model = PyNET_smaller(level=0, instance_norm=True, instance_norm_level_1=True)
#
# stat(model, (1, 1056, 1920))