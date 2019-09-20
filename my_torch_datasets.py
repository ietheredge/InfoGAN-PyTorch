import numpy as np
import torch

class Guppies(object):

    def __init__(self, dataset_zip=None, aug=True):
        loc = 'data/GuppyImages.npy'
        label_loc = 'data/GuppyLabels.npy'
        self.dataset_zip = np.load(loc)
        self.labels_zip = np.load(label_loc)
        self.aug = aug
        # self.trans = transforms.Compose([
        #     transforms.RandomRotation(5),
        #     transforms.RandomAffine(
        #         0,
        #         translate=(0.2, 0.3),
        #         scale=(0.5, 1.5),
        #         ),
        #     transforms.RandomHorizontalFlip(p=0.2)
        # ])
        # self.crop = transforms.Compose([
        #     transforms.Resize(512, interpolation=2),
        #     transforms.RandomCrop(256),
        # ])
        self.imgs = torch.from_numpy(self.dataset_zip).float()
        self.labels = self.labels_zip

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(256, 256, 3).permute(2, 0, 1)
        # if self.aug:
        #     x_t = self.trans(x)
        #     x_c = self.crop(x)
        y = self.labels[index]
        return x, y  # [x, x_t, x_c]