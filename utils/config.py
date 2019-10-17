import time
import torch
import os.path as osp
from pprint import pprint


class Config:
    # data
    dataset = "Visdrone"
    # root_dir = "/home/twsf/data/Shanghai/part_B_final"
    root_dir = '/home/twsf/data/Visdrone/'
    train_dir = osp.join(root_dir, "VisDrone2019-DET-train")
    test_dir = osp.join(root_dir, "VisDrone2019-DET-val")
    pre = '/home/twsf/work/CSRNet/run/Visdrone/experiment_1/checkpoint.path.tar'

    # train
    batch_size = 16
    input_size = (640, 480)  # (x, y)
    start_epoch = 0
    epochs = 201
    workers = 4

    log_para = 1.  # density need a factor
    downrate = 8
    gtdownrate = 8

    # param for optimizer
    lr = 0.0002
    momentum = 0.995
    decay = 5*1e-4
    steps = [0.7, 0.8, 0.9]
    scales = 0.3

    use_mulgpu = False
    gpu_id = [0, 1, 2, 3]
    device = torch.device('cuda:0')
    visualize = True
    resume = False
    print_freq = 10
    plot_every = 10  # every n batch plot
    seed = time.time()

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
