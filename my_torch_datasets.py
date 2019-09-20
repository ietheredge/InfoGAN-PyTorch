import numpy as np
import torch
import torchvision.transforms as transforms

class Guppies(object):

    def __init__(self, dataset_zip=None, aug=True):
        loc = 'data/GuppyImages.npy'
        label_loc = 'data/GuppyLabels.npy'
        self.dataset_zip = np.load(loc)
        self.labels_zip = np.load(label_loc)
        self.aug = aug
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5),
            #     (0.5, 0.5, 0.5))])
        self.imgs = torch.from_numpy(self.dataset_zip).float()
        self.labels = self.labels_zip

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(256, 256, 3).permute(2, 0, 1)
        x = self.transform(x)
        y = self.labels[index]
        return x, y  # [x, x_t, x_c]