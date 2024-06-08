import os
import random
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class RainDataset(Dataset):
    """
    Custom dataset for rain removal tasks, where each rainy image has a one-to-one correspondence with a non-rainy image.
    
    Args:
        root_dir (str): Root directory containing the 'rainy' and 'non_rainy' subdirectories.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, transform=None):
        self.rainy_dir = os.path.join(root_dir, 'rain')
        self.non_rainy_dir = os.path.join(root_dir, 'norain')
        self.images = [f[:-4] for f in os.listdir(self.rainy_dir) if f.endswith('.png')] 
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        image_name = self.images[idx]
        
        # Construct paths for rainy and non-rainy images
        rainy_path = os.path.join(self.rainy_dir, image_name + '.png')
        non_rainy_path = os.path.join(self.non_rainy_dir, image_name + '.png')  # Assuming cleaned images have '_clean' suffix
        
        # Load images
        rainy_image = Image.open(rainy_path).convert('RGB')
        non_rainy_image = Image.open(non_rainy_path).convert('RGB')
        
        if self.transform:
            rainy_image = self.transform(rainy_image)
            non_rainy_image = self.transform(non_rainy_image)
        
        return rainy_image, non_rainy_image
    

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]

class CenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = (ih - self.ch) // 2
        self.w1 = (iw - self.cw) // 2

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]

def get_files(path):
    ret = []
    path_rainy = path + "/rain"
    path_gt = path + "/norain"

    for root, dirs, files in os.walk(path_rainy):
        files.sort()    
        
        for name in files:
            #if name.split('.')[1] != 'png':
            #    continue
            file_rainy = path_rainy + "/" + name
            file_gt = path_gt + "/" + name
            ret.append([file_rainy, file_gt])
    return ret

class DenoisingDataset(Dataset):
    def __init__(self, root_dir):                                   		    # root: list ; transform: torch transform
        self.imglist = get_files(root_dir)
    def __getitem__(self, index):
        ## read an image
        # img_rainy = cv2.imread(self.imglist[index][0])
        # img_gt = cv2.imread(self.imglist[index][1])
        # img_rainy = cv2.cvtColor(img_rainy, cv2.COLOR_BGR2RGB)
        # img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_rainy = cv2.imdecode(np.fromfile(self.imglist[index][0], dtype=np.uint8), cv2.IMREAD_COLOR)
        img_gt = cv2.imdecode(np.fromfile(self.imglist[index][1], dtype=np.uint8), cv2.IMREAD_COLOR)
        img_rainy = cv2.cvtColor(img_rainy, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        cropper = CenterCrop(img_gt.shape[:2], (128, 128))
        img_rainy = cropper(img_rainy)
        img_gt = cropper(img_gt)
        # random rotate and horizontal flip

        # normalization
        img_rainy = img_rainy.astype(np.float32) # RGB image in range [0, 255]
        img_gt = img_gt.astype(np.float32) # RGB image in range [0, 255]
        img_rainy = img_rainy / 255.0
        img_rainy = torch.from_numpy(img_rainy.transpose(2, 0, 1)).contiguous()
        img_gt = img_gt / 255.0
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).contiguous()

        return img_rainy, img_gt
    
    def __len__(self):
        return len(self.imglist)
