import os
import h5py
import torch
import shutil
import numpy as np


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, is_best, epoch, filename='checkpoint_epoch{}_.pth.tar'):
    root = "work_dirs"
    if not os.path.exists(root):
        os.makedirs(root)
    file_path = os.path.join(root, filename.format(epoch))
    torch.save(state, file_path)
    if is_best:
        best_file = os.path.join(root, 'model_best.pth.tar')
        shutil.copyfile(file_path, best_file)
