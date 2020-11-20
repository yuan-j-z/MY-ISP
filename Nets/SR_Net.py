# -*- coding:utf8 -*-
import torch
import torch.nn as nn
from torch.nn import init

def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    # conv_op = nn.Conv2d(4, 3, kernel_size=3, padding=1, bias=False)
    conv_op = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)

    sobel_kernel = torch.tensor(((-1, -1, -1), (-1, 8, -1), (-1, -1, -1)), dtype=torch.float32)
    sobel_kernel = torch.reshape(sobel_kernel, (1, 1, 3, 3))
    # sobel_kernel = sobel_kernel.repeat(3, 4, 1, 1)
    sobel_kernel = sobel_kernel.repeat(4, 4, 1, 1)
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


'#######################################################################'
class Unet_all(nn.Module):
    def __init__(self, channel=32):
        super(Unet_all, self).__init__()

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

        self.up = nn.ConvTranspose2d(channel, int(channel / 4), 2, stride=2)
        self.conv10_1 = nn.Conv2d(int(channel / 4), 3, kernel_size=1, stride=1)

        self.edge = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1),
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),
            nn.Conv2d(4, 3, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

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
        out = self.conv10_1(up10)

        # edge = edge_conv2d(x)
        # edge = self.edge(edge)
        # out = edge + out

        # out = self.sigmoid(out)
        out = torch.clamp(out, 0, 1.0, out=None)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight.data)
                if m.bias:
                    init.constant(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform(m.weight.data)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             m.weight.data.normal_(0.0, 0.02)
    #             if m.bias is not None:
    #                 m.bias.data.normal_(0.0, 0.02)
    #         if isinstance(m, nn.ConvTranspose2d):
    #             m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        return torch.max(0.1 * x, x)

'#######################################################################'
class MSRB(nn.Module):
    def __init__(self, input_channels=3):
        super(MSRB, self).__init__()
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True))

        self.conv5_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True))
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True))

        self.conv_confusion = nn.Conv2d(input_channels * 4, input_channels, kernel_size=1)

    def forward(self, x):
        output3_1 = self.conv3_1(x)
        output5_1 = self.conv5_1(x)
        # cat1 = torch.cat((output3_1, output5_1), dim=1)
        cat1 = torch.cat([output3_1, output5_1], dim=1)
        output3_2 = self.conv3_2(cat1)
        output5_2 = self.conv5_2(cat1)
        # cat2 = torch.cat((output3_2, output5_2), dim=1)
        cat2 = torch.cat([output3_2, output5_2], dim=1)
        output = self.conv_confusion(cat2)
        output += x
        return output

class MSRN_edge(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(MSRN_edge, self).__init__()
        self.n_blocks = 8

        self.head = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=True))

        modules_body = nn.ModuleList()
        for i in range(self.n_blocks):
            modules_body.append(MSRB(input_channels=64))
        self.body = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(
            nn.Conv2d(64 * (self.n_blocks + 1), 64, kernel_size=1),
            nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1, bias=True),
            # nn.PixelShuffle(2),
            nn.ConvTranspose2d(64 * 4, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1, bias=True))

        self.edge = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1),
            # nn.PixelShuffle(2),
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),
            nn.Conv2d(4, 3, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, raw):
        x = self.head(raw)
        head = x

        MSRB_our = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            MSRB_our.append(x)
        MSRB_our.append(head)
        out = torch.cat(MSRB_our, dim=1)
        out = self.tail(out)

        edge = edge_conv2d(raw)
        edge = self.edge(edge)
        out = edge + out

        out = self.sigmoid(out)

        return out