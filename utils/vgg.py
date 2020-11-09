# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torchvision import models
import torch.nn as nn
import torch
import os


CONTENT_LAYER = 'relu_16'

def vgg_19():

    vgg_19 = models.vgg19(pretrained=True).features
    model = nn.Sequential()

    i = 0
    for layer in vgg_19.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            # if i == 1:
            #     r = vgg_19[0].weight[:, 0, :, :].unsqueeze(1)
            #     g = vgg_19[0].weight[:, 1, :, :].unsqueeze(1)
            #     b = vgg_19[0].weight[:, 2, :, :].unsqueeze(1)
            #     rggb = torch.cat([r, g, g, b], 1)
            #
            #     layer = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
            #     layer.weight.data = rggb
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name == CONTENT_LAYER:
            break

    # print(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    for param in model.parameters():
        param.requires_grad = False

    for param in vgg_19.parameters():
        param.requires_grad = False

    return model

def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std