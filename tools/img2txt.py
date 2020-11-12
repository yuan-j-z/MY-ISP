# -*- coding:utf8 -*-

import os
import sys
import numpy as np
import random

def img2txt():

    raw_dir = '/media/ps/2tb/yjz/data/ISP/ZZR/train/huawei_raw'
    rgb_dir = '/media/ps/2tb/yjz/data/ISP/ZZR/train/canon'
    train = open('/media/ps/2tb/yjz/MY-ISP/data/ZZR_train.txt', 'w')

    val_raw_dir = '/media/ps/2tb/yjz/data/ISP/ZZR/test/huawei_raw'
    val_rgb_dir = '/media/ps/2tb/yjz/data/ISP/ZZR/test/canon'
    val = open('/media/ps/2tb/yjz/MY-ISP/data/ZZR_val.txt', 'w')

    "# 存取训练集的数据路径"
    raw_names = os.listdir(raw_dir)
    raw_names.sort(key=lambda x: int(x.split('.')[0]))

    for raw_name in raw_names:
        raw_path = os.path.join(raw_dir, raw_name)
        rgb_path = os.path.join(rgb_dir, raw_name.split('.')[0] + '.jpg')
        train.write(raw_path + ' ' + rgb_path + '\n')

    train.close()

    "# 存取验证集的数据路径"
    val_raw_names = os.listdir(val_raw_dir)
    val_raw_names.sort(key=lambda x: int(x.split('.')[0]))

    for val_raw_name in val_raw_names:
        val_raw_path = os.path.join(val_raw_dir, val_raw_name)
        val_rgb_path = os.path.join(val_rgb_dir, val_raw_name.split('.')[0] + '.jpg')
        val.write(val_raw_path + ' ' + val_rgb_path + '\n')

    val.close()

def npy2txt():

    raw_dir = '/media/ps/2tb/yjz/data/ISP/NPY/raw-npy'
    rgb_dir = '/media/ps/2tb/yjz/data/ISP/NPY/rgb-npy'
    train = open('/media/ps/2tb/yjz/MY-ISP/data/new/train.txt', 'w')
    val = open('/media/ps/2tb/yjz/MY-ISP/data/new/val.txt', 'w')


    "# 存取训练集的数据路径"
    # 静态场景路径读取
    raw_list = [name for name in os.listdir(raw_dir)
                  if name != 'case10' and name != "case20" and name != "case30" and name != "case62"]
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
                      if name == 'case10' or name == "case20" or name == "case30" or name == "case62"]
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
    img2txt()
    # npy2txt()