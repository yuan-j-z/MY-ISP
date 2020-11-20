# -*- coding: utf-8 -*-

import os
import av
import cv2
import sys
import numpy as np

def yuv2bgr(video_dir, height, width, startfrm):
    """
    :param filename: 待处理 YUV 视频的名字
    :param height: YUV 视频中图像的高
    :param width: YUV 视频中图像的宽
    :param startfrm: 起始帧
    :return: None
    """
    """A dataset for loading filelist as.
        like::
            data/case1/1.jpg
            data/case1/2.jpg
            data/case1/N.jpg
            data/case2/1.jpg
            data/case2/2.jpg
            data/case2/N.jpg
        """
    caselist = os.listdir(video_dir)
    caselist.sort(key=lambda x: int(x.split('e')[1]))
    # 转换test文件下的yuv时关闭下面这句话
    # caselist.sort(key=lambda x: int(x.split('_')[0]))

    for case in caselist:
        filepath = os.path.join(video_dir, case)
        print(filepath)
        filenames = os.listdir(filepath)
        filenames.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))
        print(filenames)
        for filename in filenames:
            fp = open(os.path.join(filepath, filename), 'rb')

            # YUV420是RGB24内存的一半
            framesize = height * width * 3 // 2  # 一帧图像所含的像素个数
            h_h = height // 2
            h_w = width // 2

            fp.seek(0, 2)  # 设置文件指针到文件流的尾部
            ps = fp.tell()  # 当前文件指针位置
            numfrm = ps // framesize  # 计算输出帧数
            fp.seek(framesize * startfrm, 0)

            for i in range(numfrm - startfrm):
                Yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
                Ut = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
                Vt = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')

                for m in range(height):
                    for n in range(width):
                        # ord以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值
                        Yt[m, n] = ord(fp.read(1))
                for m in range(h_h):
                    for n in range(h_w):
                        Ut[m, n] = ord(fp.read(1))
                for m in range(h_h):
                    for n in range(h_w):
                        Vt[m, n] = ord(fp.read(1))

                img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))

                img = img.reshape((height * 3 // 2, width)).astype('uint8')  # YUV 的存储格式为：NV12（YYYY UV）

                # 由于 opencv 不能直接读取 YUV 格式的文件, 所以要转换一下格式
                bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)  # 注意 YUV 的存储格式

                rgb_dir = os.path.join(os.path.dirname(video_dir), "rgb")
                save_dir = os.path.join(rgb_dir, "{}".format(case))
                # save_dir = "/media/ps/2tb/yjz/data/our_data/test/Temporary/rgb/{}".format(case)
                if not os.path.exists(rgb_dir):
                    os.makedirs(rgb_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, '{}.png'.format(filename.split('.')[0]))

                cv2.imwrite(save_path, bgr_img)
            fp.close()
    print("finished")
    return None

# 将一个文件夹下的每一个文件一一保存到npy文件中
def oneimg2npy_(root_dir, out_path):
    caselist = os.listdir(root_dir)
    caselist.sort(key=lambda x: int(x.split('e')[1]))
    for case in caselist:
        print("\nstart ", case)
        filelist = os.listdir(os.path.join(root_dir, case))

        '# 读取raw数据时使用'
        filelist.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))
        for file in filelist:
            bayer = np.zeros(shape=2073600, dtype='uint16')
            file_path = os.path.join(root_dir, case, file)

            with open(file_path, "rb") as f:
                for i in range(0, len(f.read()), 2):
                    f.seek(i)
                    raw = f.read(2)
                    a1 = int((raw[0] / 16) % 16)
                    a2 = int(raw[0] % 16)
                    a3 = int((raw[1] / 16) % 16)
                    a4 = int(raw[1] % 16)
                    value = a3 * 256 + a4 * 16 + a1 * 1
                    bayer[int(i / 2)] = value

            bayer = bayer.reshape(1080, 1920)

            height, width = bayer.shape[0], bayer.shape[1]
            # print(height, width)
            h = height // 2
            w = width // 2

            R = np.zeros(shape=(h, w), dtype='uint16', order='C')
            Gr = np.zeros(shape=(h, w), dtype='uint16', order='C')
            Gb = np.zeros(shape=(h, w), dtype='uint16', order='C')
            B = np.zeros(shape=(h, w), dtype='uint16', order='C')

            for x in range(height):
                for y in range(0, width, 2):
                    if x % 2 == 0:
                        R[int(x / 2)][int(y / 2)] = bayer[x][y]
                        Gr[int(x / 2)][int(y / 2)] = bayer[x][y + 1]
                    elif x % 2 == 1:
                        Gb[int(x / 2)][int(y / 2)] = bayer[x][y]
                        B[int(x / 2)][int(y / 2)] = bayer[x][y + 1]

            out = np.stack((R, Gr, Gb, B))

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            out_dir = os.path.join(out_path, case)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            np.save(os.path.join(out_dir, './{}.npy'.format(file.split(".")[0])), out)
            print("saved: {}".format(file))

            f.close()

        '# 读取rgb数据时使用'
        # filelist.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))
        #
        # i = 0
        # for file in filelist:
        #     file_path = os.path.join(root_dir, case, file)
        #     label = cv2.imread(file_path)
        #     label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        #     label = np.transpose(label, (2, 0, 1))
        #
        #     save_path = os.path.join(out_path, case)
        #     if not os.path.exists(out_path):
        #         os.makedirs(out_path)
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     np.save(os.path.join(save_path, './{}.npy'.format(file.split(".")[0])), label)
        #
        #     i += 1
        #     print("第{}张转换完成！ ".format(i))

        print("{} finised".format(case))

    print("All cases had finished!")

if __name__ == '__main__':
    # _ = yuv2bgr(video_dir="D:\\DATA\\raw-yuv\\new\\yuv", height=1080, width=1920, startfrm=0)

    _ = oneimg2npy_(root_dir='/media/ps/2tb/yjz/data/ISP/NPY/test_raw',
                    out_path="/media/ps/2tb/yjz/data/ISP/NPY/test-raw-npy")
