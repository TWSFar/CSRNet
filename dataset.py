import random
from torch.utils.data import Dataset
from image import load_data


class listDataset(Dataset):
    def __init__(self,
                 root,
                 shape=None,
                 transform=None,
                 train=False,
                 batch_size=1):
        # if train:
        #     root = root * 4
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.batch_size = batch_size

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        # assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        img, target = load_data(img_path, self.train)
        # img = 255.0 * F.to_tensor(img)
        # img[0,:,:]=img[0,:,:]-92.8207477031
        # img[1,:,:]=img[1,:,:]-95.2757037428
        # img[2,:,:]=img[2,:,:]-104.877445883
        if self.transform is not None:
            img = self.transform(img)

        return img, target
