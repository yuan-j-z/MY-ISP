# -*- coding:utf8 -*-

import os
import sys
import numpy as np
import random

def img2txt():

    raw = '/media/ps/2tb/yjz/data/our_data/npy/train/raw'
    # rgb = '/media/ps/2tb/yjz/data/our_data/npy/train/yuv'     # RGB 图像转 NPY 时将其分辨率从1080*1920 resize 到 540*960
    rgb = '/media/ps/2tb/yjz/data/our_data/npy/train/1080_1920_yuv'
    train = open('/media/ps/2tb/yjz/MY-ISP/data/train.txt', 'w')

    val_raw = '/media/ps/2tb/yjz/data/our_data/npy/val/raw'
    # val_rgb = '/media/ps/2tb/yjz/data/our_data/npy/val/yuv'    # RGB 图像转 NPY 时将其分辨率从1080*1920 resize 到 540*960
    val_rgb = '/media/ps/2tb/yjz/data/our_data/npy/val/1080_1920_yuv'
    val = open('/media/ps/2tb/yjz/MY-ISP/data/val.txt', 'w')

    "# 存取训练集的数据路径"
    raw_names = os.listdir(raw)
    raw_names.sort(key=lambda x: int(x.split('_')[0]))

    for raw_name in raw_names:
        raw_filenames = os.listdir(os.path.join(raw, raw_name))
        raw_filenames.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))
        for raw_filename in raw_filenames:
            raw_path = os.path.join(raw, raw_name, raw_filename)
            train.write(raw_path + '\n')

    rgb_names = os.listdir(rgb)
    rgb_names.sort(key=lambda x: int(x.split('_')[0]))
    for rgb_name in rgb_names:
        rgb_filenames = os.listdir(os.path.join(rgb, rgb_name))
        rgb_filenames.sort(key=lambda x: int((x.split('_', 5)[5]).split('.')[0]), reverse=True)
        for rgb_filename in rgb_filenames:
            rgb_path = os.path.join(rgb, rgb_name, rgb_filename)
            train.write(rgb_path + '\n')

    train.close()

    "# 存取验证集的数据路径"
    val_raw_names = os.listdir(val_raw)
    val_raw_names.sort(key=lambda x: int(x.split('_')[0]))

    for val_raw_name in val_raw_names:
        val_raw_filenames = os.listdir(os.path.join(val_raw, val_raw_name))
        val_raw_filenames.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))
        for val_raw_filename in val_raw_filenames:
            val_raw_path = os.path.join(val_raw, val_raw_name, val_raw_filename)
            val.write(val_raw_path + '\n')

    val_rgb_names = os.listdir(val_rgb)
    val_rgb_names.sort(key=lambda x: int(x.split('_')[0]))
    for val_rgb_name in val_rgb_names:
        val_rgb_filenames = os.listdir(os.path.join(val_rgb, val_rgb_name))
        val_rgb_filenames.sort(key=lambda x: int((x.split('_', 5)[5]).split('.')[0]), reverse=True)
        for val_rgb_filename in val_rgb_filenames:
            val_rgb_path = os.path.join(val_rgb, val_rgb_name, val_rgb_filename)
            val.write(val_rgb_path + '\n')

    val.close()

def npy2txt():

    raw_dir = '/media/ps/2tb/yjz/data/ISP/NPY/raw-npy'
    rgb_dir = '/media/ps/2tb/yjz/data/ISP/NPY/rgb-npy'
    train = open('/media/ps/2tb/yjz/MY-ISP/data/new/train.txt', 'w')
    val = open('/media/ps/2tb/yjz/MY-ISP/data/new/val.txt', 'w')


    "# 存取训练集的数据路径"
    # 静态场景路径读取
    raw_list = [name for name in os.listdir(raw_dir)
                  if name != 'case10' and name != "case20" and name != "case30"]
    raw_list.sort(key=lambda x: int(x.split('e')[1]))

    for case in raw_list:
        raw_filelist = os.listdir(os.path.join(raw_dir, case))
        raw_filelist.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))

        rgb_filelist = os.listdir(os.path.join(rgb_dir, case))
        rgb_filelist.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))

        for i in range (len(raw_filelist)):
            raw_file = raw_filelist[i]
            raw_path = os.path.join(raw_dir, case, raw_file)
            rgb_file = rgb_filelist[i]
            rgb_path = os.path.join(rgb_dir, case, rgb_file)
            train.write(raw_path + ' ' + rgb_path + '\n')

    train.close()

    "# 存取验证集的数据路径"
    # 静态场景路径读取
    val_raw_list = [name for name in os.listdir(raw_dir)
                      if name == 'case10' or name == "case20" or name == "case30"]
    val_raw_list.sort(key=lambda x: int(x.split('e')[1]))

    for val_case in val_raw_list:
        val_raw_filelist = os.listdir(os.path.join(raw_dir, val_case))
        val_raw_filelist.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))

        val_rgb_filelist = os.listdir(os.path.join(rgb_dir, val_case))
        val_rgb_filelist.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))

        for i in range(len(val_raw_filelist)):
            val_raw_file = val_raw_filelist[i]
            val_raw_path = os.path.join(raw_dir, val_case, val_raw_file)
            val_rgb_file = val_rgb_filelist[i]
            val_rgb_path = os.path.join(rgb_dir, val_case, val_rgb_file)
            val.write(val_raw_path + ' ' + val_rgb_path + '\n')

    val.close()

    print("Finished!!!")

if __name__ == "__main__":
    # img2txt()
    npy2txt()