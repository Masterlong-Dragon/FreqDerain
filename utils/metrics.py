import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np
from math import log10

def calculate_psnr(img1, img2):
    """
    计算两幅图像的PSNR。
    :param img1: 第一幅图像，Tensor类型
    :param img2: 第二幅图像，Tensor类型
    :return: PSNR值
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # 假设图像是归一化的，最大像素值为1
    psnr = 20 * log10(max_pixel / torch.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2, data_range=1.0):
    """
    计算两幅图像的SSIM。
    :param img1: 第一幅图像，Tensor类型，需要转换为numpy
    :param img2: 第二幅图像，Tensor类型，需要转换为numpy
    :param data_range: 数据范围，默认为1.0（归一化图像）
    :return: SSIM值
    """
    img1 = img1.cpu().numpy().transpose((1, 2, 0))  # 转换为HWC
    img2 = img2.cpu().numpy().transpose((1, 2, 0))
    ssim_value = ssim(img1, img2, data_range=data_range)
    return ssim_value
