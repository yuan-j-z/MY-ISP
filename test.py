# -*- coding: utf-8 -*-

import torch
import os, cv2
import time
import numpy as np
from Nets.SR_Net import *
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

class TEST_DATA(Dataset):
    def __init__(self, test_dir, count, H, W):

        self.test_path = []
        # self.H, self.W = H, W
        self.H, self.W = int(H / 2), int(W / 2)

        test_caselist = os.listdir(test_dir)
        # test_caselist.sort(key=lambda x: int(x.split('e')[1]))
        print("there are {} cases".format(len(test_caselist)))

        for case in test_caselist:
            test_files = os.listdir(os.path.join(test_dir, case))
            test_files.sort(key=lambda x: int((x.split('_', 5)[5]).split('_')[0]))
            for test_file in test_files[:count]:
                test_file_dir = os.path.join(test_dir, case, test_file)
                self.test_path.append(test_file_dir)

    def __len__(self):
        print("test dataset--------->len=", len(self.test_path))
        return len(self.test_path)

    def __getitem__(self, index):
        test_raw = np.load(self.test_path[index])

        '######################################################################################'
        # img = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(img, (3, 3), 0)
        # gaussImg = cv2.Canny(blurred, 50, 100)
        # gaussImg = np.expand_dims(gaussImg, axis=2)
        # gaussImg = np.repeat(gaussImg, 3, 2)
        # test_image = cv2.fastNlMeansDenoisingColored(test_image, None, 5, 5, 7, 21)
        # test_image = test_image + gaussImg * 5


        source = (test_raw / 4095.).astype(np.float32)

        '# Unet的设计导致要32对齐'
        _, h, w = source.shape  # H W C
        if w % 16 != 0:
            w = (w // 16) * 16
        if h % 16 != 0:
            h = (h // 16) * 16
        source = source[:, :h, :w]

        '# saving the input'
        source_a = (source * 255.).astype(np.uint8)
        cv2.imwrite(save_path + "/" + "orig_" + str(index + 1) + ".jpg", source_a[0])    # h265: .jpg

        source = torch.from_numpy(source)

        return source


def test(test_dir, count, H, W, model, weights_dir, save_path):
    test_data = TEST_DATA(test_dir=test_dir, count=count, H=H, W=W)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)


    state_dict = torch.load(weights_dir)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, source in enumerate(test_loader):

            source = source.cuda()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                test_output = model(source)
            torch.cuda.synchronize()
            end = time.time()
            print("test time:", (end - start) / 100)
            print("FPS: %f" % (1.0 / ((end - start) / 100)))

            test_output = test_output.permute(0, 2, 3, 1).cpu().data.numpy()
            test_output = np.minimum(np.maximum(test_output, 0), 1)
            print(test_output.shape)
            test_output = (test_output[0, ...] * 255.).astype(np.uint8)
            print(test_output.shape)
            test_output = cv2.cvtColor(test_output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path + '/' + str(idx + 1) + '.jpg', test_output)
            print("saving the {} images\n".format(str(idx + 1)))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    test_dir = '/media/ps/2tb/yjz/data/ISP/NPY/test_ray-npy'
    weights_dir = './model/Unet_SR/weights/best_psnr/best_psnr.pt'
    save_path = './model/Unet_SR/results'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)


    count, H, W = 1, 1080, 1920
    model = Unet_SR(channel=32, is_train=False)
    test(test_dir, count, H, W, model, weights_dir, save_path)