#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import rawpy
import cv2, os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class supervision_dataset(Dataset):
    def __init__(self, root_dir, crop_size, for_train=True):

        self.crop_size = crop_size

        self.imgs = []
        self.labels = []
        caselist = os.listdir(root_dir)

        if for_train == True:
            for case in caselist:
                if case == "raw":
                    raw_names = os.listdir(os.path.join(root_dir, case))
                    raw_names.sort(key=lambda x: int(x.split('_')[0]))
                    for raw_name in raw_names:
                        raw_filenames = os.listdir(os.path.join(root_dir, case, raw_name))
                        raw_filenames.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))
                        raw_filelist = [os.path.join(root_dir, case, raw_name, raw_filename)
                                        for raw_filename in raw_filenames]
                        self.imgs.extend(raw_filelist)
                elif case == "yuv":
                    yuv_names = os.listdir(os.path.join(root_dir, case))
                    yuv_names.sort(key=lambda x: int(x.split('_')[0]))
                    for yuv_name in yuv_names:
                        yuv_filenames = os.listdir(os.path.join(root_dir, case, yuv_name))
                        yuv_filenames.sort(key=lambda x: int((x.split('_', 5)[5]).split('.')[0]), reverse=True)
                        yuv_filelist = [os.path.join(root_dir, case, yuv_name, yuv_filename)
                                        for yuv_filename in yuv_filenames]
                        self.labels.extend(yuv_filelist)
        else:
            for case in caselist:
                if case == "raw":
                    raw_names = os.listdir(os.path.join(root_dir, case))
                    raw_names.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))
                    raw_list = [os.path.join(root_dir, case, raw_name) for raw_name in raw_names]
                    self.imgs.extend(raw_list)
                elif case == "yuv":
                    yuv_names = os.listdir(os.path.join(root_dir, case))
                    yuv_names.sort(key=lambda x: int((x.split('_', 5)[5]).split('.')[0]), reverse=True)
                    yuv_list = [os.path.join(root_dir, case, yuv_name) for yuv_name in yuv_names]
                    self.labels.extend(yuv_list)

    def __len__(self):
        """Returns length of dataset."""
        print("Dataset--------->len=", len(self.imgs))
        return len(self.imgs)

    def __getitem__(self, index):
        # index = 1
        img = np.load(self.imgs[index])
        label = np.load(self.labels[index])

        h, w = img.shape[1], img.shape[2]
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)
        img = img[:, i:i + self.crop_size, j:j + self.crop_size]
        source = img / 4095.0

        label = label[:, i:i + self.crop_size, j:j + self.crop_size]
        label = np.transpose(label, (1, 2, 0))

        label2 = cv2.resize(label, (label.shape[1] // 2, label.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        label3 = cv2.resize(label, (label.shape[1] // 4, label.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
        label4 = cv2.resize(label, (label.shape[1] // 8, label.shape[0] // 8), interpolation=cv2.INTER_NEAREST)
        label5 = cv2.resize(label, (label.shape[1] // 16, label.shape[0] // 16), interpolation=cv2.INTER_NEAREST)

        target1 = np.transpose(label, (2, 0, 1)) / 255.0
        target2 = np.transpose(label2, (2, 0, 1)) / 255.0
        target3 = np.transpose(label3, (2, 0, 1)) / 255.0
        target4 = np.transpose(label4, (2, 0, 1)) / 255.0
        target5 = np.transpose(label5, (2, 0, 1)) / 255.0


        source = torch.from_numpy(source)
        target1 = torch.from_numpy(target1)
        target2 = torch.from_numpy(target2)
        target3 = torch.from_numpy(target3)
        target4 = torch.from_numpy(target4)
        target5 = torch.from_numpy(target5)

        return source, target1, target2, target3, target4, target5
        
class raw_dataset(Dataset):
    def __init__(self, root_dir, crop_size, is_train=True):
        
        self.crop_size = crop_size
        self.is_train = is_train
        self.data = []
        self.raw = []
        self.rgb = []
        
        with open(root_dir, "rb") as f:
            for line in f.readlines():
                self.data.append(line[:-1])

        if is_train:
            self.raw = self.data[:4370]
            self.rgb = self.data[4370:]
        else:
            self.raw = self.data[:800]
            self.rgb = self.data[800:]

    def __len__(self):
        """Returns length of dataset."""
        print("Dataset--------->len=", len(self.raw))
        return len(self.raw)
        
    def __getitem__(self, index):
        raw = np.load(self.raw[index])
        rgb = np.load(self.rgb[index])

        raw = (raw / 4095.).astype(np.float32)
        rgb = (rgb / 255.).astype(np.float32)

        _, h, w = raw.shape
        if self.is_train:
            "# 随机裁剪patch送入模型(self.crop_size, self.crop_size)"
            i = torch.randint(0, h - self.crop_size + 1, (1,)).item()
            j = torch.randint(0, w - self.crop_size + 1, (1,)).item()
            source = raw[:, i:i + self.crop_size, j:j + self.crop_size]
            target = rgb[:, i * 2:(i + self.crop_size) * 2, j * 2:(j + self.crop_size) * 2]
        else:
            i = (h - self.crop_size) // 2
            j = (w - self.crop_size) // 2
            source = raw[:, i:i + self.crop_size * 4, j:j + self.crop_size * 4]
            target = rgb[:, i * 2:(i + self.crop_size * 4) * 2, j * 2:(j + self.crop_size * 4) * 2]
        
        if self.is_train:
            '随机翻转'
            if np.random.random() > 0.5:
                source = np.flip(source, axis=0)
                target = np.flip(target, axis=0)
            if np.random.random() > 0.5:
                source = np.flip(source, axis=1)
                target = np.flip(target, axis=1)
            if np.random.random() > 0.5:
                source = np.rot90(source, 2)
                target = np.rot90(target, 2)
            source = source.copy()
            target = target.copy()

        # raw = (raw * 255.).astype(np.uint8)
        # cv2.imwrite("raw.jpg", raw[0])
        #
        # rgb = np.transpose(rgb, (1, 2, 0))
        # rgb = (rgb * 255.).astype(np.uint8)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("rgb.jpg", rgb)
        #
        # source = (source * 255.).astype(np.uint8)
        # cv2.imwrite("source.jpg", source[0])
        #
        # target = np.transpose(target, (1, 2, 0))
        # target = (target * 255.).astype(np.uint8)
        # target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("target.jpg", target)

        source = torch.from_numpy(source)
        target = torch.from_numpy(target)

        return source, target

class isp_dataset(Dataset):
    def __init__(self, root_dir, crop_size, is_train=True):

        self.crop_size = crop_size
        self.is_train = is_train
        self.raw, self.rgb = [], []

        with open(root_dir, "rb") as f:
            for line in f.readlines():
                a = ' '.encode()
                self.raw.append(line.split(a)[0])
                self.rgb.append(line.split(a)[1][:-1])

    def __len__(self):
        """Returns length of dataset."""
        print("Dataset--------->len=", len(self.raw))
        return len(self.raw)

    def __getitem__(self, index):

        raw = np.load(str(self.raw[index], encoding="utf-8"))
        rgb = np.load(str(self.rgb[index], encoding="utf-8"))

        _, h, w = raw.shape
        rgb_half = cv2.resize(np.transpose(rgb, (1, 2, 0)), (w, h), interpolation=cv2.INTER_CUBIC)
        rgb_half = np.transpose(rgb_half, (2, 0, 1))

        raw = (raw / 4095.).astype(np.float32)
        rgb = (rgb / 255.).astype(np.float32)
        rgb_half = (rgb_half / 255.).astype(np.float32)

        if self.is_train:
            "# 随机裁剪patch送入模型(self.crop_size, self.crop_size)"
            i = torch.randint(0, h - self.crop_size + 1, (1,)).item()
            j = torch.randint(0, w - self.crop_size + 1, (1,)).item()
            source = raw[:, i:i + self.crop_size, j:j + self.crop_size]
            rgb_half = rgb_half[:, i:i + self.crop_size, j:j + self.crop_size]
            target = rgb[:, i * 2:(i + self.crop_size) * 2, j * 2:(j + self.crop_size) * 2]
        else:
            i = (h - self.crop_size) // 2
            j = (w - self.crop_size) // 2
            source = raw[:, i:i + self.crop_size * 4, j:j + self.crop_size * 4]
            rgb_half = rgb_half[:, i:i + self.crop_size * 4, j:j + self.crop_size * 4]
            target = rgb[:, i * 2:(i + self.crop_size * 4) * 2, j * 2:(j + self.crop_size * 4) * 2]

        # raw = (raw * 255.).astype(np.uint8)
        # cv2.imwrite("raw.jpg", raw[0])
        #
        # rgb = np.transpose(rgb, (1, 2, 0))
        # rgb = (rgb * 255.).astype(np.uint8)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("rgb.jpg", rgb)
        #
        # source = (source * 255.).astype(np.uint8)
        # cv2.imwrite("source.jpg", source[0])
        #
        # rgb_half = np.transpose(rgb_half, (1, 2, 0))
        # rgb_half = (rgb_half * 255.).astype(np.uint8)
        # rgb_half = cv2.cvtColor(rgb_half, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("rgb_half.jpg", rgb_half)
        #
        # target = np.transpose(target, (1, 2, 0))
        # target = (target * 255.).astype(np.uint8)
        # target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("target.jpg", target)

        if self.is_train:
            '随机翻转'
            if np.random.random() > 0.5:
                source = np.flip(source, axis=0)
                rgb_half = np.flip(rgb_half, axis=0)
                target = np.flip(target, axis=0)
            if np.random.random() > 0.5:
                source = np.flip(source, axis=1)
                rgb_half = np.flip(rgb_half, axis=1)
                target = np.flip(target, axis=1)
            if np.random.random() > 0.5:
                source = np.rot90(source, 2)
                rgb_half = np.rot90(rgb_half, 2)
                target = np.rot90(target, 2)
            source = source.copy()
            rgb_half = rgb_half.copy()
            target = target.copy()

        source = torch.from_numpy(source)
        rgb_half = torch.from_numpy(rgb_half)
        target = torch.from_numpy(target)

        return source, rgb_half, target

# if __name__ == "__main__":
#
#     import torch.nn as nn
#     def edge_conv2d(im):
#         # 用nn.Conv2d定义卷积操作
#         conv_op = nn.Conv2d(4, 3, kernel_size=3, padding=1, bias=False)
#
#         sobel_kernel = torch.tensor(((-1, -1, -1), (-1, 8, -1), (-1, -1, -1)), dtype=torch.float32)
#         sobel_kernel = torch.reshape(sobel_kernel, (1, 1, 3, 3))
#         sobel_kernel = sobel_kernel.repeat(3, 4, 1, 1)
#         conv_op.weight.data = sobel_kernel.cuda()
#         edge_detect = conv_op(im)
#
#         return edge_detect
#
#     root_dir = '/media/ps/2tb/yjz/MY-ISP/data/new/val.txt'
#     dataset = isp_dataset(root_dir=root_dir, crop_size=64, is_train=False)
#     loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
#
#     for idx, (source, target) in enumerate(loader):
#         print(idx)
#
#         source = source.cuda(non_blocking=True)
#         out = edge_conv2d(source)
#
#         output = out.permute(0, 2, 3, 1).cpu().data.numpy()
#         output = np.minimum(np.maximum(output, 0), 1)
#         output = (output[0, :, :, :] * 255.).astype(np.uint8)
#         output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
#         cv2.imwrite("out2.jpg", output_bgr)
#         print(out.shape)
#
#         # print(source.shape, H_lr.shape, target.shape)
#         # print(list[0].shape, list[1].shape, list[2].shape, list[3].shape)
