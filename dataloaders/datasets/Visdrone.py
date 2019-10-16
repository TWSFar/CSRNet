import os
import cv2
import math
import h5py
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
from dataloaders.noMaskTransforms import transfrom


class VisdroneDataset(Dataset):
    def __init__(self, data_dir, train=True):
        super().__init__()
        self.data_dir = data_dir

        self.img_dir = osp.join(self.data_dir, 'images')
        self.img_list = [file for file in os.listdir(self.img_dir)]
        self.train = train
        self.img_number = len(self.img_list)

        # transform
        self.stf = transfrom(self.train)

    def __getitem__(self, index):
        img_path = osp.join(self.img_dir, self.img_list[index])
        gt_path = img_path.replace('.jpg', '.png').replace('images', 'DensityMask')
        assert osp.isfile(img_path), '{} not exist'.format(img_path)
        assert osp.isfile(gt_path), '{} not exist'.format(gt_path)

        img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
        target = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        o_h, o_w = img.shape[:2]

        img, target = self.stf(img, target)
        scale = torch.tensor([o_h / img.shape[1],
                              o_w / img.shape[2]])
        return img, target, scale

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = VisdroneDataset("/home/twsf/data/Visdrone/VisDrone2018-DET-val", train=False)
    data = dataset.__getitem__(0)
    dataloader = DataLoader(dataset, batch_size=2)
    for (img, label, scale) in dataloader:
        pass
    pass
