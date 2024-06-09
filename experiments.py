from matplotlib import pyplot as plt
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
# transforms.append(CropWithResize(config.crop_size))
transforms.append(ToTensor())
# transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
transforms = Compose(transforms)
# 加载数据集
train_dataset = DenoisingDataset(root_dir=config.train_data_dir)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

train_dataset2 = RainDataset(root_dir=config.train_data_dir, transform=transforms, crop=CropWithResize(config.crop_size))
train_loader2 = DataLoader(train_dataset2, batch_size=config.batch_size, shuffle=False)

# 比较二者归一化的方式
# rainy_image, non_rainy_image = train_dataset[0]
rainy_image2, non_rainy_image2 = train_loader2.dataset[0]
rainy_image22, non_rainy_image22 = train_loader2.dataset[0]
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(rainy_image2.permute(1, 2, 0).numpy())
plt.title("Rainy Image")
plt.subplot(1, 2, 2)
plt.imshow(rainy_image22.permute(1, 2, 0).numpy())
plt.title("Rainy Image")
plt.show()
# # 比较属性
# print(rainy_image.shape, non_rainy_image.shape)
# print(rainy_image2.shape, non_rainy_image2.shape)
# # 比较值
# print(rainy_image[0, 0, 0], rainy_image2[0, 0, 0])
# print(rainy_image[0, 0, 1], rainy_image2[0, 0, 1])
# print(rainy_image[0, 0, 2], rainy_image2[0, 0, 2])

# 显示train_loader的数据
# for i, data in enumerate(train_loader2):
#     rainy_image, non_rainy_image = data
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(rainy_image[0].permute(1, 2, 0).numpy())
#     plt.title("Rainy Image")
#     plt.subplot(1, 2, 2)
#     plt.imshow(non_rainy_image[0].permute(1, 2, 0).numpy())
#     plt.title("Non Rainy Image")
#     plt.show()