import cv2
import random
import numpy as np

import torch
import sys
import os.path as osp
import torchvision.transforms as stf
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from utils.config import opt


# ===============================img tranforms============================

class RandomHorizontallyFlip(object):
    def __call__(self, img, den):
        x_flip = random.choice([True, False])
        if x_flip:
            img = img[:, ::-1, :]
            den = den[:, ::-1]

        return img, den


class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (w, h)

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        return img


class DeNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# ===============================label tranforms============================


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor*self.para
        return tensor


class transfrom(object):
    def __init__(self, train=True):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.train = train
        self.img_resize = FreeScale(opt.input_size)
        self.flip = RandomHorizontallyFlip()
        self.lab_factor = LabelNormalize(opt.log_para)
        self.img_normalize = stf.Compose([
            stf.ToTensor(),
            stf.Normalize(self.mean, self.std)])

    def __call__(self, img, den):
        if self.train:
            img, den = self.flip(img, den)

        img = self.img_resize(img)
        img = self.img_normalize(img)
        den = self.lab_factor(den)

        return img, den
