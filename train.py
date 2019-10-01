import os
import json
import time
from tqdm import tqdm
import argparse
import visdom

import dataset
from visualization import create_vis_plot, update_vis_plot
from model import CSRNet
from utils import save_checkpoint

import torch
import torch.nn as nn
from torchvision import transforms
import multiprocessing
multiprocessing.set_start_method('spawn', True)


parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--train_json', metavar='TRAIN',
                    default="/home/twsf/work/CSRNet/part_A_train.json",
                    help='path to train json')
parser.add_argument('--test_json', metavar='TEST',
                    default="/home/twsf/work/CSRNet/part_A_test.json",
                    help='path to test json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,
                    type=str, help='path to the pretrained model')
parser.add_argument('--gpu', metavar='GPU', type=str, default="GPU",
                    help='GPU id to use.')
parser.add_argument('--visdom', type=bool, default=True,
                    help='use visdom.')

args = parser.parse_args()
args.original_lr = 1e-7
args.lr = 1e-7
args.batch_size = 60
args.momentum = 0.95
args.decay = 5*1e-4
args.start_epoch = 0
args.epochs = 400
args.steps = [-1, 1, 100, 150]
args.scales = [1, 1, 1, 1]
args.workers = 4
args.seed = time.time()
args.print_freq = 30
args.use_mulgpu = True
args.gpu_id = [0, 1, 2]
device = torch.device('cuda:0')
root = "/home/twsf/data/Shanghai/"
vis = visdom.Visdom()


def path_transform(imgs_list):
    new_list = []
    for line in imgs_list:
        dirs = line.split('/')[-4:]
        line = root
        for dir in dirs:
            line = os.path.join(line, dir)
        new_list.append(line)
    return new_list


def main():
    best_prec1 = 1e6
    # Visdom
    if args.visdom:
        vis_legend = ["Loss", "MAE"]
        batch_plot = create_vis_plot(vis, 'Batch', 'Loss', 'batch loss', vis_legend[0:1])
        val_plot = create_vis_plot(vis, 'Epoch', 'result', 'val result', vis_legend[1:2])

    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    train_list = path_transform(train_list)

    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)
    val_list = path_transform(val_list)

    # Dataset dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(
            train_list,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]),
            train=True,
            batch_size=args.batch_size),
        num_workers=args.workers,
        shuffle=True,
        batch_size=1)   # must be 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(
                val_list,
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])]),
                train=False),
        batch_size=1)  # must be 1, because per image size is different

    torch.cuda.manual_seed(args.seed)

    model = CSRNet()
    model = model.to(device)
    if args.use_mulgpu:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    criterion = nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, batch_plot)
        prec1 = validate(test_loader, model, criterion, epoch, val_plot)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        if epoch % 20 == 0 and epoch != 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.module.state_dict() if args.use_mulgpu
                else model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.module.state_dict() if args.use_mulgpu
                else model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch)


def train(train_loader, model, criterion, optimizer, epoch, batch_plot):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    print('epoch %d, processed %d samples, lr %.10f' %
          (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.to(device)
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(0).to(device)

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        # visdom
        if args.visdom:
            update_vis_plot(vis, i, [loss.cpu().tolist()], batch_plot, 'append')


def validate(test_loader, model, criterion, epoch, val_plot):
    print('begin val')
    model.eval()
    mae = 0
    for i, (img, target) in enumerate(tqdm(test_loader)):
        img = img.to(device)
        output = model(img)
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(device))
    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '.format(mae=mae))

    # visdom
    if args.visdom:
        update_vis_plot(vis, epoch, [mae], val_plot, 'append')

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


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
