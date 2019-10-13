import os
import fire
import time
from tqdm import tqdm
# import visdom

from dataloaders.dataset import SHTDataset
from utils.visualization import TensorboardSummary
from model import CSRNet
from utils.saver import Saver
from utils.config import opt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Trainer(object):
    def __init__(self):
        self.best_pred = 1e6

        # Define Saver
        self.saver = Saver(opt)
        self.saver.save_experiment_config()

        # visualize
        if opt.visualize:
            # vis_legend = ["Loss", "MAE"]
            # batch_plot = create_vis_plot(vis, 'Batch', 'Loss', 'batch loss', vis_legend[0:1])
            # val_plot = create_vis_plot(vis, 'Epoch', 'result', 'val result', vis_legend[1:2])
            # Define Tensorboard Summary
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        # Dataset dataloader
        self.train_dataset = SHTDataset(opt.train_dir, train=True)
        self.train_loader = DataLoader(
            self.train_dataset,
            num_workers=opt.workers,
            shuffle=True,
            batch_size=opt.batch_size)   # must be 1
        self.test_dataset = SHTDataset(opt.test_dir, train=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=opt.batch_size)  # must be 1, because per image size is different

        torch.cuda.manual_seed(opt.seed)

        model = CSRNet()
        self.model = model.to(opt.device)
        if opt.use_mulgpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=opt.gpu_id)
        self.criterion = nn.MSELoss(reduction='mean').to(opt.device)
        self.optimizer = torch.optim.SGD(self. model.parameters(), opt.lr,
                                         momentum=opt.momentum,
                                         weight_decay=opt.decay)
        if opt.resume:
            if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                opt.start_epoch = checkpoint['epoch']
                self.best_pred = checkpoint['best_pred']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.pre, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pre))

    def train(self, epoch):
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        num_img_tr = len(self.train_loader)
        print('epoch %d, processed %d samples, lr %.10f' %
              (epoch, epoch * len(self.train_loader.dataset), opt.lr))

        start = time.time()

        for i, (img, target, _)in enumerate(self.train_loader):
            img = img.to(opt.device)
            target = target.type(torch.FloatTensor).unsqueeze(1).to(opt.device)

            output = self.model(img)

            loss = self.criterion(output, target)
            losses.update(loss.item(), img.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)

            # visualize
            # if opt.visualize:
            #     update_vis_plot(vis, i, [loss.cpu().tolist()], batch_plot, 'append')
            global_step = i + num_img_tr * epoch
            self.writer.add_scalar('train/total_loss_epoch', loss.cpu().item(), global_step)
            if (i + 1) % opt.plot_every == 0:
                self.summary.visualize_image(self.writer, opt.dataset, img, target, output, global_step)

            if i % opt.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      .format(
                        epoch, i, len(self.train_loader),
                        batch_time=batch_time,
                        data_time=data_time, loss=losses))

    def validate(self, epoch):
        print('begin val')
        self.model.eval()
        mae = 0
        for i, (img, target, scale) in enumerate(tqdm(self.test_loader)):
            img = img.to(opt.device)
            output = self.model(img)
            mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(opt.device))

        mae = mae / self.test_dataset.img_number

        # visualize
        # if opt.visualize:
        #     update_vis_plot(vis, epoch, [mae], val_plot, 'append')
        self.writer.add_scalar('val/total_loss_epoch', mae, epoch)
        print(' * MAE {mae:.3f} '.format(mae=mae))

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


def train(**kwargs):
    opt._parse(kwargs)
    trainer = Trainer()
    for epoch in range(opt.start_epoch, opt.epochs):
        adjust_learning_rate(trainer.optimizer, epoch)

        # train
        trainer.model.train()
        trainer.train(epoch)

        # val
        prec1 = trainer.validate(epoch)

        is_best = prec1 < trainer.best_pred
        trainer.best_pred = min(prec1, trainer.best_pred)
        print(' * best MAE {mae:.3f} '.format(mae=trainer.best_pred))
        if (epoch % 20 == 0 and epoch != 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': trainer.model.module.state_dict() if opt.use_mulgpu
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)


if __name__ == '__main__':
    fire.Fire()
    # train()
