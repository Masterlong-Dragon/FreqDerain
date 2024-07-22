from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from models.derain_model import DerainModel
from utils.metrics import calculate_psnr
from utils.config import Config
from data.dataset import DenoisingDataset, RainDataset, Rain100LDataset
from utils.transforms import CropWithResize

from PIL import Image

import os
import numpy as np


import losses.pytorch_ssim as pytorch_ssim
criterion_ssim = pytorch_ssim.SSIM().cuda()

# 数据预处理
transforms = []
# transforms.append(CropWithResize(config.crop_size, False))
transforms.append(ToTensor())
transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
transforms = Compose(transforms)

mean = torch.tensor([0.5, 0.5, 0.5])  # 每个通道的均值
std = torch.tensor([0.5, 0.5, 0.5])   # 每个通道的标准差


def validate(model, dataloader, device):
    """
    验证模型性能。
    """
    
    origin_folder = r"F:\EIProject\derain\output\origin"
    pred_folder = r"F:\EIProject\derain\output\pred"
    
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            inputs = [inp.to(device) for inp in inputs]
            outputs = model(inputs)
            
            ssim = criterion_ssim(outputs, inputs[1])
            ssim_values.append(ssim)
            
            # 恢复原始尺度
            outputs = outputs.cpu() * std.view(3, 1, 1) + mean.view(3, 1, 1)
            inputs = [inp.cpu() * std.view(3, 1, 1) + mean.view(3, 1, 1) for inp in inputs]
            
            original = np.clip(inputs[0].squeeze().permute(1, 2, 0).numpy(), 0, 1)
            pred = np.clip(outputs.squeeze().permute(1, 2, 0).numpy(), 0, 1)
            
            psnr = calculate_psnr(outputs, inputs[1])
            psnr_values.append(psnr)
            
            
            plt.imsave(os.path.join(origin_folder, f'image_{i}.png'), original)
            plt.imsave(os.path.join(pred_folder, f'image_{i}.png'), pred)
            # # BGR to RGB
            # original = inputs[0].squeeze().permute(1, 2, 0).numpy()
            # pred = outputs.squeeze().permute(1, 2, 0).numpy()
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(original)
            # plt.title("Rainy Image")
            # plt.axis("off")
            # plt.subplot(1, 2, 2)
            # plt.imshow(pred)
            # plt.show()
            
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    print(f"Average SSIM on validation set: {avg_ssim:.4f}")
    return avg_psnr

def preprocess_image(image, tile_size=(256, 256), stride=256):
    w, h = image.size
    tiles = []
    i = 0
    j = 0
    while i < h - tile_size[1] + 1:
        while j < w - tile_size[0] + 1:
            tile = image.crop((j, i, j + tile_size[0], i + tile_size[1]))
            tile_tensor = transforms(tile).unsqueeze(0)
            tiles.append((tile_tensor, i, j))
            j += stride
        # 如果j尚且未到达边界，那么最后一个tile的宽度不足tile_size[0]
        # 向左移动一个stride，以保证最后一个tile的宽度为tile_size[0]
        if j < w:
            j = w - tile_size[0]
            tile = image.crop((j, i, j + tile_size[0], i + tile_size[1]))
            tile_tensor = transforms(tile).unsqueeze(0)
            tiles.append((tile_tensor, i, j))
        i += stride
        j = 0
    # 如果i尚且未到达边界，那么最后一行的高度不足tile_size[1]
    # 向上移动一个stride，以保证最后一行的高度为tile_size[1]
    if i < h:
        i = h - tile_size[1]
        j = 0
        while j < w - tile_size[0] + 1:
            tile = image.crop((j, i, j + tile_size[0], i + tile_size[1]))
            tile_tensor = transforms(tile).unsqueeze(0)
            tiles.append((tile_tensor, i, j))
            j += stride
        # 如果j尚且未到达边界，那么最后一个tile的宽度不足tile_size[0]
        # 向左移动一个stride，以保证最后一个tile的宽度为tile_size[0]
        if j < w:
            j = w - tile_size[0]
            tile = image.crop((j, i, j + tile_size[0], i + tile_size[1]))
            tile_tensor = transforms(tile).unsqueeze(0)
            tiles.append((tile_tensor, i, j))
    return tiles

def stitch_tiles(tiles, height, width, tile_size=(256, 256), stride=256):
    result = np.zeros((height, width, 3))
    # 与process_image的流程类似
    index = 0
    i = 0
    j = 0
    while i < height - tile_size[1] + 1:
        while j < width - tile_size[0] + 1:
            tile = tiles[index][0].squeeze().permute(1, 2, 0).numpy()
            result[i:i+tile_size[1], j:j+tile_size[0], :] = tile
            index += 1
            j += stride
        if j < width:
            j = width - tile_size[0]
            tile = tiles[index][0].squeeze().permute(1, 2, 0).numpy()
            result[i:i+tile_size[1], j:j+tile_size[0], :] = tile
            index += 1
        i += stride
        j = 0
    if i < height:
        i = height - tile_size[1]
        j = 0
        while j < width - tile_size[0] + 1:
            tile = tiles[index][0].squeeze().permute(1, 2, 0).numpy()
            result[i:i+tile_size[1], j:j+tile_size[0], :] = tile
            index += 1
            j += stride
        if j < width:
            j = width - tile_size[0]
            tile = tiles[index][0].squeeze().permute(1, 2, 0).numpy()
            result[i:i+tile_size[1], j:j+tile_size[0], :] = tile
            index += 1
    return result

def process_images_in_folder(folder_path, save_path, model, device):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                tiles = preprocess_image(img)
                processed_tiles = []
                for tile, i, j in tiles:
                    tile = tile.to(device)
                    with torch.no_grad():
                        output = model([tile, None]).cpu() * std.view(3, 1, 1) + mean.view(3, 1, 1)
                    processed_tiles.append((output, i, j))
                stitched_result = stitch_tiles(processed_tiles, img.height, img.width)
                stitched_result = np.clip(stitched_result, 0, 1)
                plt.imsave(os.path.join(save_path, f"derained_{filename}"), stitched_result)
                



def main():
    config = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载验证数据集
    # 注意：这里假设验证数据集包含雨天图像及其对应的干净图像
    val_dataset = RainDataset(root_dir=config.test_data_dir, transform=transforms, crop=CropWithResize(config.crop_size, False))
    val_loader = DataLoader(val_dataset, batch_size=config.tbatch_size, shuffle=False)
    
    # 加载训练好的模型
    model_path = r"C:\Users\18379\Downloads\best_model.pth" # 假定这是最佳模型的路径
    model = DerainModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # model.load(model_path)
    
    # # 计算模型参数总量
    # total_params = sum(p.numel() for p in model.gen.parameters())
    # print(f"Total number of parameters: {total_params}")
    sums_all = 0
    for name, param in model.gen.named_children():
        print(f"Layer: {name}")
        params_count = sum(p.numel() for p in param.parameters())
        sums_all += params_count
        print(f"Number of parameters: {params_count}\n")
    
    print(f"Total number of parameters: {sums_all}")
    # # 如果你想要分别计算可训练和不可训练的参数量，可以使用下面的代码：
    # total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total_params_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # print(f"Total number of trainable parameters: {total_params_trainable}")
    # print(f"Total number of non-trainable parameters: {total_params_non_trainable}")
    
    # 验证模型
    # avg_psnr = validate(model, val_loader, device)
    # print(f"Average PSNR on validation set: {avg_psnr:.2f} dB")
    # process_images_in_folder(os.path.join(config.test_data_dir, 'rain'), model, device)

if __name__ == "__main__":
    main()