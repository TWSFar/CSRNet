import h5py
import math
import random
import numpy as np
from PIL import Image


def BiBubic(x):
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x**2) + (x**3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x**2) - (x**3)
    else:
        return 0


def BiCubic_interpolation(input, out_scale=(30, 40)):
    scrH, scrW = input.shape
    dstH, dstW = out_scale
    output = np.zeros(out_scale)
    ratio_x = (scrH / dstH)
    ratio_y = (scrW / dstW)
    for i in range(dstH):
        for j in range(dstW):
            scrx = i * ratio_x
            scry = j * ratio_y
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x+ii < 0 or y+jj < 0 or x+ii >= scrH or y+jj >= scrW:
                        continue
                    tmp += input[x+ii, y+jj] * BiBubic(ii-u) * BiBubic(jj-v)
            output[i, j] = tmp

    return output


def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    if train:
        crop_size = (int(img.size[0]/2), int(img.size[1]/2))
        if random.randint(0, 9) <= -1:
            dx = int(random.randint(0, 1)*img.size[0]*1./2)
            dy = int(random.randint(0, 1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        img = img.crop((dx, dy, crop_size[0]+dx, crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy, dx:crop_size[0]+dx]
        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # print(target.shape)
    target = BiCubic_interpolation(
        target,
        (int(target.shape[0]/8), int(target.shape[1]/8))) * 64

    return img, target


if __name__ == '__main__':
    load_data("/home/twsf/data/Shanghai/part_A_final/train_data/images/IMG_1.jpg")
