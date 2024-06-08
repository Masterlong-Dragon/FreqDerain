import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from data.dataset import DenoisingDataset, RainDataset

from models.derain_model import DerainModel
from utils.config import Config
from utils.logger import Logger
from utils.transforms import CropWithResize 



config = Config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = []
transforms.append(CropWithResize(config.crop_size, False))
transforms.append(ToTensor())
# transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
transforms = Compose(transforms)
# 加载数据集
train_dataset = DenoisingDataset(root_dir=config.train_data_dir)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

train_dataset2 = RainDataset(root_dir=config.train_data_dir, transform=transforms)
train_loader2 = DataLoader(train_dataset2, batch_size=config.batch_size, shuffle=False)

# 比较二者归一化的方式
rainy_image, non_rainy_image = train_dataset[0]
rainy_image2, non_rainy_image2 = train_dataset2[0]
# 比较属性
print(rainy_image.shape, non_rainy_image.shape)
print(rainy_image2.shape, non_rainy_image2.shape)
# 比较值
print(rainy_image[0, 0, 0], rainy_image2[0, 0, 0])
print(rainy_image[0, 0, 1], rainy_image2[0, 0, 1])
print(rainy_image[0, 0, 2], rainy_image2[0, 0, 2])