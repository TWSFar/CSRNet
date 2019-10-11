import time
import torch
import os.path as osp


class Config:
    # data
    root_dir = "/home/twsf/data/Shanghai/part_B_final"
    train_dir = osp.join(root_dir, "train_data")
    test_dir = osp.join(root_dir, "test_data")
    pre = '/home/twsf/work/CSRNet/work_dirs/SHT_B_model_best.pth.tar'

    # train
    batch_size = 6
    input_size = (768, 576)  # (x, y)
    start_epoch = 0
    epochs = 200
    workers = 8
    mean = [0.452016860247, 0.447249650955, 0.431981861591]
    std = [0.23242045939, 0.224925786257, 0.221840232611]
    log_para = 100.  # density need a factor
    downrate = 8
    gtdownrate = 8

    # param for optimizer
    original_lr = 1e-5
    lr = 1e-5
    momentum = 0.995
    decay = 5*1e-4
    steps = [-1, 1, 100, 150]

    scales = [1, 1, 1, 1]
    seed = time.time()
    print_freq = 30

    use_mulgpu = False
    gpu_id = [0, 1, 2]
    device = torch.device('cuda:0')
    visualize = True
    resume = False
    plot_every = 1  # every 1 batch plot


opt = Config()
