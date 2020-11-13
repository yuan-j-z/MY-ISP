#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, os
from torch import nn
from Nets.SR_Net import *
from Nets.PYNET import *
from utils.ssim import *
from collections import OrderedDict


def selet_model(outtype, channel=64):
    if outtype == 'Unet_SR':
        model = Unet_SR(channel=channel, is_train=True)
    elif outtype == 'Unet_SR_small':
        model = Unet_SR_small(channel=channel, is_train=True)
    elif outtype == 'Unet_all':
        model = Unet_all()
    else:
        assert False, 'Not supported outtype:{}'.format(outtype)
    print("\033[32musing %s \033[0m" % (outtype))

    return model

def selet_level_model(outtype, level, channel=64):
    if outtype == 'PyNET_smaller':
        model = PyNET_smaller(level=level, channel=channel, instance_norm=True, instance_norm_level_1=True)
    elif outtype == 'SE_ResNet':
        model = SE_ResNet(level=level, channel=channel)
    else:
        assert False, 'Not supported outtype:{}'.format(outtype)
    print("\033[32musing %s \033[0m" % (outtype + "_level" + str(level)))

    return model


def selet_loss(loss='l1'):
    if loss == 'l1':
        LOSS = nn.L1Loss()
    elif loss == 'l2':
        LOSS = nn.MSELoss()
    elif loss == 'ssim':
        LOSS = SSIM_LOSS(data_range=1.0)
    elif loss == 'l1_ssim':
        LOSS = MIX_L1_SSIM_LOSS(data_range=1.0)
    else:
        assert False, 'Using unsupported loss'

    return LOSS

def check_load_weights(model, weights_path, use_cuda=True):

    print('Loading checkpoint from: {}'.format(weights_path))

    if use_cuda:
        model.load_state_dict(torch.load(weights_path), strict=False)
    else:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)

def load_weights(model, level, model_path, use_cuda=True):

    weights_path = os.path.join(model_path, 'level{}'.format(level + 1), 'weights')
    ckpt_fname = os.path.join(weights_path, 'best_psnr', 'best_psnr.pt')
    print('Loading checkpoint from: {}'.format(ckpt_fname))

    if use_cuda:
        model.load_state_dict(torch.load(ckpt_fname), strict=False)
    else:
        model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'), strict=False)

    print(model.load_state_dict(torch.load(ckpt_fname), strict=False))

def print_params(params, train_loader):
    print('Training parameters: ')
    param_dict = vars(params)
    pretty = lambda x: x.replace('_', ' ').capitalize()
    print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
    print()

    num_batches = len(train_loader)
    if params.report_interval == 0:
        params.report_interval = num_batches
    print("--------->num_batches:", num_batches)
    print("--------->report_interval:", params.report_interval)
    assert num_batches % params.report_interval == 0, 'Report interval must divide total number of batches'

    return num_batches