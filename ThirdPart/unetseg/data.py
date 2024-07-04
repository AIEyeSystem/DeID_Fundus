import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image 

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.size = (512,512)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.size)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
class DriveDataset2(Dataset):
    def __init__(self, images_path, masks_path,img_transform=None,mask_transform=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.size = (512,512)
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        """ Reading image """
        image = Image.open(self.images_path[index])
        # image = image.convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)

        """ Reading mask """
        mask = Image.open(self.masks_path[index])
        mask = mask.convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return self.n_samples