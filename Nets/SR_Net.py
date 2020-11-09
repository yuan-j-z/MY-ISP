# -*- coding:utf8 -*-
import torch
import torch.nn as nn

def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(4, 3, kernel_size=3, padding=1, bias=False)

    sobel_kernel = torch.tensor(((-1, -1, -1), (-1, 8, -1), (-1, -1, -1)), dtype=torch.float32)
    sobel_kernel = torch.reshape(sobel_kernel, (1, 1, 3, 3))
    sobel_kernel = sobel_kernel.repeat(3, 4, 1, 1)
    conv_op.weight.data = sobel_kernel.cuda()
    edge_detect = conv_op(im)

    return edge_detect

class Unet_SR(nn.Module):
    def __init__(self, channel=32, is_train=True):
        super(Unet_SR, self).__init__()
        self.is_train = is_train

        self.conv1_1 = nn.Conv2d(4, channel, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(channel, channel * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(channel * 8, channel * 8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(channel * 8, channel * 16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(channel * 16, channel * 16, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(channel * 16, channel * 8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(channel * 16, channel * 8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(channel * 8, channel * 8, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(channel * 8, channel * 4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(channel * 8, channel * 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(channel * 4, channel * 2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(channel * 4, channel * 2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(channel * 2, channel, 2, stride=2)
        self.conv9_1 = nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        # 残差学习
        self.end = nn.Conv2d(channel, 4, kernel_size=3, padding=1, stride=1)
        self.end_out = nn.Conv2d(4, 3, kernel_size=3, padding=1, stride=1)

        self.up = nn.ConvTranspose2d(channel, int(channel / 4), 2, stride=2)
        self.conv10_1 = nn.Conv2d(int(channel / 4), 3, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        up10 = self.up(conv9)
        SR = self.conv10_1(up10)
        SR = torch.clamp(SR, 0, 1.0, out=None)

        if self.is_train:
            end = self.end(conv9)
            end_out = self.end_out(end + x)

            # # using the edge
            # edge = edge_conv2d(x)
            # end_out = end_out + edge * 0.5

            end_out = torch.clamp(end_out, 0, 1.0, out=None)

            return SR, end_out
        else:
            return SR

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        return torch.max(0.1 * x, x)

class Unet_SR_student(nn.Module):
    def __init__(self, channel=32, is_train=True):
        super(Unet_SR_student, self).__init__()
        self.is_train = is_train

        self.conv1_1 = nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(channel, channel * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(channel * 4, channel * 2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(channel * 4, channel * 2, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(channel * 2, channel, 2, stride=2)
        self.conv9_1 = nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1)

        self.L_sr = nn.Conv2d(channel, 3, kernel_size=3, padding=1, stride=1)

        # self.up = nn.PixelShuffle(2)
        # self.conv10_1 = nn.Conv2d(int(channel / 4), 3, kernel_size=1, stride=1)
        self.up = nn.ConvTranspose2d(channel, int(channel / 4), 2, stride=2)
        self.conv10_1 = nn.Conv2d(int(channel / 4), 3, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))

        up8 = self.upv8(conv3)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))

        up10 = self.up(conv9)
        SR = self.conv10_1(up10)
        SR = torch.clamp(SR, 0, 1.0, out=None)

        return SR

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        return torch.max(0.1 * x, x)

class Unet_SR_small(nn.Module):
    def __init__(self, channel=32, is_train=True):
        super(Unet_SR_small, self).__init__()
        self.is_train = is_train

        self.conv1_1 = nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4)

        self.conv2_1 = nn.Conv2d(channel, channel * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(channel * 2, channel, 2, stride=4, output_padding=2)
        self.conv9_1 = nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        self.L_sr = nn.Conv2d(channel, 3, kernel_size=3, padding=1, stride=1)

        # self.up = nn.PixelShuffle(2)
        # self.conv10_1 = nn.Conv2d(int(channel / 4), 3, kernel_size=1, stride=1)
        self.up = nn.ConvTranspose2d(channel, int(channel / 4), 2, stride=2)
        self.conv10_1 = nn.Conv2d(int(channel / 4), 3, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))

        up9 = self.upv9(conv2)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))

        up10 = self.up(conv9)
        SR = self.conv10_1(up10)
        SR = torch.clamp(SR, min=0.0, max=1.0, out=None)

        return SR

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        return torch.max(0.1 * x, x)