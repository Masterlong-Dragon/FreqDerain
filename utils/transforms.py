import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, Pad, ToPILImage
import random
import torch.nn.functional as F
import cv2
from torchvision.transforms.functional import crop

class CropWithResize(object):
    def __init__(self, size, random_crop=True):
        self.size = (size, size)
        self.random_crop = random_crop

    def get_params(self, img):
        w, h = img.size
        th, tw = self.size

        # If the image dimensions are already larger than the target crop size,
        # we can directly use the random/center cropping logic.
        if w >= tw and h >= th:
            if self.random_crop:
                i = random.randint(0, h - th)
                j = random.randint(0, w - tw)
            else:
                i = (h - th) // 2
                j = (w - tw) // 2
        else:
            # Calculate the scaling factor based on the shortest side
            scale_factor = max(tw / w, th / h)
            resized_width = int(w * scale_factor)
            resized_height = int(h * scale_factor)
            
            # Resize the image before cropping
            img = F.resize(img, (resized_height, resized_width))

            # After resizing, the image should now have dimensions >= crop size
            if self.random_crop:
                i = random.randint(0, resized_height - th) if resized_height > th else 0
                j = random.randint(0, resized_width - tw) if resized_width > tw else 0
            else:
                i = (resized_height - th) // 2
                j = (resized_width - tw) // 2

        return crop(img, i, j, th, tw)

    def __call__(self, img):
        return self.get_params(img)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, random_crop={1})'.format(self.size, self.random_crop)