import os
import json
import time
from tqdm import tqdm
import visdom

from dataloaders.dataset import SHTDataset
from dataloaders.transforms import re_tsf
from utils.visualization import create_vis_plot, update_vis_plot
from model import CSRNet
from utils.utils import save_checkpoint
from utils.config import opt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing
multiprocessing.set_start_method('spawn', True)
vis = visdom.Visdom()


def main():
    best_prec1 = 1e6

    # visualize
    if opt.visualize:
        vis_legend = ["Loss", "MAE"]
        batch_plot = create_vis_plot(vis, 'Batch', 'Loss', 'batch loss', vis_legend[0:1])
        val_plot = create_vis_plot(vis, 'Epoch', 'result', 'val result', vis_legend[1:2])

    # Dataset dataloader
    train_dataset = SHTDataset(opt.train_dir, train=True)
    train_loader = DataLoader(
        train_dataset,
        num_workers=opt.workers,
        shuffle=True,
        batch_size=opt.batch_size)   # must be 1
    test_dataset = SHTDataset(opt.test_dir, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=opt.batch_size)  # must be 1, because per image size is different

    torch.cuda.manual_seed(opt.seed)

    model = CSRNet()
    model = model.to(opt.device)
    if opt.use_mulgpu:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_id)
    criterion = nn.MSELoss(reduction='mean').to(opt.device)
    optimizer = torch.optim.SGD(model.parameters(), opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.decay)
    if opt.resume:
        if os.path.isfile(opt.pre):
            print("=> loading checkpoint '{}'".format(opt.pre))
            checkpoint = torch.load(opt.pre)
            opt.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.pre))

    for epoch in range(opt.start_epoch, opt.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, batch_plot)
        prec1 = validate(test_loader, model, criterion, epoch, val_plot)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        if epoch % 40 == 0 and epoch != 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if opt.use_mulgpu
                else model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if opt.use_mulgpu
                else model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch)


def train(train_loader, model, criterion, optimizer, epoch, batch_plot):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    print('epoch %d, processed %d samples, lr %.10f' %
          (epoch, epoch * len(train_loader.dataset), opt.lr))

    model.train()
    end = time.time()

    for i, (img, target, _)in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.to(opt.device)

        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(1).to(opt.device)

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        # visualize
        if opt.visualize:
            update_vis_plot(vis, i, [loss.cpu().tolist()], batch_plot, 'append')

            if (i + 1) % opt.batch_size == 0:
                pass


def validate(test_loader, model, criterion, epoch, val_plot):
    print('begin val')
    model.eval()
    mae = 0
    for i, (img, target, scale) in enumerate(tqdm(test_loader)):
        img = img.to(opt.device)
        output = model(img)
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(opt.device))
    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '.format(mae=mae))

    # visualize
    if opt.visualize:
        update_vis_plot(vis, epoch, [mae], val_plot, 'append')

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    opt.lr = opt.original_lr
    for i in range(len(opt.steps)):
        scale = opt.scales[i] if i < len(opt.scales) else 1
        if epoch >= opt.steps[i]:
            opt.lr = opt.lr * scale
            if epoch == opt.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
