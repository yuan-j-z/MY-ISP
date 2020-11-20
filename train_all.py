#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import json
import datetime
from apex import amp
from torchsummary import summary
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, lr_scheduler

from Nets.SR_Net import *
import dataset
from utils.ssim import *
from utils.utils import *
from utils.vgg import *
from utils.augments import *
from utils.train_tools import *


def parse_args():
    """Command-line argument parser for training."""

    '# New parser'
    parser = ArgumentParser(description='PyTorch implementation of denoise from Yuan. (2020)')

    "dataset"
    parser.add_argument('-t', '--train-dir', help='train path', default='./data/train_part.txt')
    parser.add_argument('-v', '--valid-dir', help='val path', default='./data/val_part.txt')
    parser.add_argument('-n', '--data-type', help='data type', choices=['rgbdata'], default='rgbdata', type=str)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=224, type=int)

    "augmentations"
    parser.add_argument("--use_moa", action="store_true", default=True)
    parser.add_argument("--augs", nargs="*", default=["None"])
    parser.add_argument("--prob", nargs="*", default=[1.0])
    parser.add_argument("--alpha", nargs="*", default=[1.0])
    parser.add_argument("--mix_p", nargs="*")
    parser.add_argument("--aux_prob", type=float, default=1.0)
    parser.add_argument("--aux_alpha", type=float, default=1.2)

    "models"
    parser.add_argument('-o', '--outtype', help='output type', choices=['Unet_all'],
                        default='Unet_all', type=str)
    parser.add_argument('-C', '--channel', help='the input of channel', default=32, type=int)
    parser.add_argument('--model-path', help='model save path', default='./model/Unet_part')

    "training setups"
    parser.add_argument('-l', '--loss', choices=['l1', 'l2'], default='l1', type=str)
    parser.add_argument('--workers-num', help='workers-num', default=8, type=int)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=16, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=2000, type=int)
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=1e-4, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    # 使用混合精度时打开
    # parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-3], type=list)


    "misc"
    parser.add_argument('--half', default=False)
    parser.add_argument('--apex', default=False)
    parser.add_argument('--environ', default="1", type=str)
    parser.add_argument('--teacher', default=None, type=str)
    parser.add_argument('--resume-ckpt',  default=None, type=str)
    # parser.add_argument('--resume-ckpt',  default='./model/Unet_SR-1/weights/best_psnr/best_psnr.pt', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true', default=True)
    parser.add_argument('--report-interval', help='batch report interval', default=0, type=int)
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true', default=True)

    return parser.parse_args()

def save_epoch_result(epoch, idx, target, val_output, teacher_out=None, train=False):
    if (epoch + 1) % 100 == 0 and (idx + 1) % 2 == 0:
        if train:
            path1 = os.path.join(params.model_path, 'train_result')
            path2 = os.path.join(path1, 'epoch' + str(epoch + 1))
            if not os.path.isdir(params.model_path):
                os.mkdir(params.model_path)
            if not os.path.isdir(path1):
                os.mkdir(path1)
            if not os.path.isdir(path2):
                os.mkdir(path2)
        else:
            path1 = os.path.join(params.model_path, 'val_result')
            path2 = os.path.join(path1, 'epoch' + str(epoch + 1))
            if not os.path.isdir(path1):
                os.mkdir(path1)
            if not os.path.isdir(path2):
                os.mkdir(path2)

        label = target.permute(0, 2, 3, 1).cpu().data.numpy()
        label = np.minimum(np.maximum(label, 0), 1)
        label = (label[0, :, :, :] * 255.).astype(np.uint8)
        label_bgr = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)

        output = val_output.permute(0, 2, 3, 1).cpu().data.numpy()
        output = np.minimum(np.maximum(output, 0), 1)
        output = (output[0, :, :, :] * 255.).astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        if params.teacher:
            teacher_SR = teacher_out.permute(0, 2, 3, 1).cpu().data.numpy()
            teacher_SR = np.minimum(np.maximum(teacher_SR, 0), 1)
            teacher_SR = (teacher_SR[0, :, :, :] * 255.).astype(np.uint8)
            teacher_SR = cv2.cvtColor(teacher_SR, cv2.COLOR_RGB2BGR)

            temp = np.concatenate((label_bgr, teacher_SR, output_bgr), axis=1)
        else:
            temp = np.concatenate((label_bgr, output_bgr), axis=1)

        cv2.imwrite(path2 + '/' + str(idx + 1) + '.jpg', temp)


def train_model():

    global teacher_model, teacher_out

    torch.backends.cudnn.deterministic = True
    print("CUDA visible devices: " + str(torch.cuda.device_count()))

    train_data = dataset.isp_all_dataset(params.train_dir, crop_size=params.crop_size)
    train_loader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True, num_workers=params.workers_num, pin_memory=True, drop_last=True)

    valid_data = dataset.isp_all_dataset(params.valid_dir, crop_size=params.crop_size, is_train=False)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    model = selet_model(outtype=params.outtype, channel=params.channel)
    LOSS = selet_loss(loss=params.loss)

    VGG_19 = vgg_19()

    optim = Adam(model.parameters(), lr=params.learning_rate, betas=params.adam[:2], eps=params.adam[2])
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=params.nb_epochs / 5, factor=0.1, verbose=True)

    use_cuda = torch.cuda.is_available() and params.cuda
    if use_cuda:
        LOSS = LOSS.cuda()
        model = model.cuda()
        if not params.apex:
            model = torch.nn.DataParallel(model)
        summary(model, (4, params.crop_size, params.crop_size))

        if params.teacher:
            teacher_model = Unet_SR(channel=params.channel)
            teacher_model = teacher_model.cuda()
            teacher_model = torch.nn.DataParallel(teacher_model)

            model.load_state_dict(torch.load(params.teacher), strict=False)
            teacher_model.load_state_dict(torch.load(params.teacher), strict=True)
            print("loading the teacher model weights!")
            print(model.load_state_dict(torch.load(params.teacher), strict=False))
            print(teacher_model.load_state_dict(torch.load(params.teacher), strict=True))

        if params.resume_ckpt:
            model.load_state_dict(torch.load(params.resume_ckpt), strict=False)
            print("loading the check point weights!")
            print(model.load_state_dict(torch.load(params.resume_ckpt), strict=False))

    if params.apex:
        model, optimizer = amp.initialize(model, optim, opt_level="O1")  # 这里是“欧一”
        # model = DDP(model, delay_allreduce=True)

    if params.half:
        model = model.half()
        print('\n\033[35m使用混合精度训练！！！\033[0m')

    num_batches = print_params(params, train_loader)

    stats = {'data_type': params.data_type,
             'train_loss': [],
             'valid_loss': [],
             'valid_psnr': []}

    if params.teacher:
        teacher_model.eval()

    print("\033[34mStart Training! \033[0m")
    best_psnr = 0
    train_start = datetime.now()
    for epoch in range(params.nb_epochs):

        torch.cuda.empty_cache()
        print('\n\033[36mEPOCH {:d} / {:d}\033[0m'.format(epoch + 1, params.nb_epochs))

        epoch_start = datetime.now()
        train_loss_meter = AvgMeter()
        loss_meter = AvgMeter()
        time_meter = AvgMeter()

        model.train()
        for batch_idx, (source, target) in enumerate(train_loader):
            batch_start = datetime.now()
            progress_bar(batch_idx, num_batches, params.report_interval, loss_meter.val)

            if use_cuda:
                source = source.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            if params.teacher:
                teacher_out = teacher_model(source)

            if params.half:
                source = source.half()
                target = target.half()

            '########################数据增强#########################'
            # "# match the resolution of (LR, HR) due to CutBlur"
            # if target.size() != source.size():
            #     scale = target.size(2) // source.size(2)
            #     source = F.interpolate(source, scale_factor=scale, mode="nearest")
            #
            # if params.use_moa:
            #     params.augs = ["blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]
            #     params.prob = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            #     params.alpha = [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.3]
            #     params.aux_prob, params.aux_alpha = 1.0, 1.2
            #     params.mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]
            #
            #     if params.outtype == 'DIDN':
            #         params.prob = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
            #     else:
            #         params.prob = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
            #
            # target, source, mask, aug = apply_augment(target, source,
            #     params.augs, params.prob, params.alpha,
            #     params.aux_prob, params.aux_alpha, params.mix_p)
            #
            # # a = np.minimum(np.maximum(source.permute(0, 2, 3, 1).cpu().data.numpy(), 0), 1)[0, :, :, :] * 255.0
            # # b = np.minimum(np.maximum(target.permute(0, 2, 3, 1).cpu().data.numpy(), 0), 1)[0, :, :, :] * 255.0
            # # a, b = a.astype(np.uint8), b.astype(np.uint8)
            # # cv2.imwrite("a.png", a[:, :, 0]), cv2.imwrite("b.png", b[:, :, 0])
            '#########################################################'

            out = model(source)

            '########################数据增强#########################'
            # if aug == "cutout":
            #     output, target = output * mask, target * mask
            '#########################################################'

            if params.teacher:
                loss = LOSS(out, target) + LOSS(out, teacher_out) * 0.5
            else:
                loss = LOSS(out, target)
            loss_meter.update(loss.item())

            optim.zero_grad()
            if params.apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()

            if params.teacher:
                save_epoch_result(epoch, batch_idx, target, out, teacher_out=teacher_out, train=True)
            else:
                save_epoch_result(epoch, batch_idx, target, out, train=True)

            # Report/update statistics
            time_meter.update(time_elapsed_since(batch_start)[1])
            if (batch_idx + 1) % params.report_interval == 0 and batch_idx:
                show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                train_loss_meter.update(loss_meter.avg)
                loss_meter.reset()
                time_meter.reset()

        # Evaluate model on validation set
        print('\n' '\rTesting model on validation set... ' '\n', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        model.eval()

        valid_start = datetime.now()
        psnr_meter = AvgMeter()
        loss_val_meter = AvgMeter()

        with torch.no_grad():
            for batch_idx, (val_source, val_target) in enumerate(valid_loader):
                if use_cuda:
                    val_source = val_source.cuda(non_blocking=True)
                    val_target = val_target.cuda(non_blocking=True)

                if params.teacher:
                    val_teacher_out = teacher_model(val_source)

                if params.half:
                    val_source = val_source.half()
                    val_target = val_target.half()

                val_out = model(val_source)

                if params.teacher:
                    val_loss = LOSS(val_out, val_target) + LOSS(val_out, val_teacher_out) * 0.5
                else:
                    val_loss = LOSS(val_out, val_target)
                loss_val_meter.update(val_loss.item())

                psnr_output = torch.clamp(val_out, 0, 1).data.cpu()
                psnr_target = torch.clamp(val_target, 0, 1).data.cpu()
                _psnr = get_psnr(psnr_output.numpy(), psnr_target.numpy())
                psnr_meter.update(_psnr, 1)

                if params.teacher:
                    save_epoch_result(epoch, batch_idx, val_target, val_out, teacher_out=val_teacher_out)
                else:
                    save_epoch_result(epoch, batch_idx, val_target, val_out)

        valid_loss = loss_val_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg

        print("loss: %.4f, psnr: %.4f" % (valid_loss, psnr_avg))

        show_on_epoch_end(epoch_time, valid_time, valid_loss, psnr_avg)
        scheduler.step(valid_loss)

        '保存权重，最优权重和json文件'
        stats['train_loss'].append(train_loss_meter.avg)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(psnr_avg)

        ckpt_dir = os.path.join(params.model_path, 'weights')
        if not os.path.isdir(params.model_path):
            os.mkdir(params.model_path)
        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)

        # # 保存权重
        # valid_loss = stats['valid_loss'][epoch]
        # fname = '{}/epoch{}-{:>1.5f}.pt'.format(ckpt_dir, epoch + 1, valid_loss)
        # print('Saving checkpoint to: {}\n'.format(fname))
        # torch.save(model.state_dict(), fname)
        #
        # # 保存最优权重
        if best_psnr <= stats['valid_psnr'][epoch]:
            best_psnr = stats['valid_psnr'][epoch]
            best_psnr_path = os.path.join(ckpt_dir, 'best_psnr')
            if not os.path.isdir(best_psnr_path):
                os.mkdir(best_psnr_path)
            psnr_name = '{}/best_psnr.pt'.format(best_psnr_path)
            print("\033[32mThe best psnr is: %s \033[0m" % '{:.2f}'.format(best_psnr))
            print("Saving best psnr to: {}".format(psnr_name))
            torch.save(model.state_dict(), psnr_name)

        # Save stats to JSON
        fname_dict = '{}/stats.json'.format(ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

        if params.plot_stats:
            loss_str = '{} loss'.format(params.loss.upper())

            plot_path1 = os.path.join(params.model_path, 'result_plt')
            if not os.path.isdir(plot_path1):
                os.mkdir(plot_path1)
            plot_per_epoch(plot_path1, 'Train loss', stats['train_loss'], loss_str)
            plot_per_epoch(plot_path1, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(plot_path1, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')
        train_loss_meter.reset()

    train_elapsed = time_elapsed_since(train_start)[0]
    print('Training done! Total elapsed time: {}\n'.format(train_elapsed))


if __name__ == '__main__':
    # 固定随机种子
    np.random.seed(0)
    torch.manual_seed(0)  # cpu设置随机种子
    torch.cuda.manual_seed_all(0)  # 为所有gpu设置随机种子

    params = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = params.environ
    train_model()