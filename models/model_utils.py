from typing import List
import ptwt
import torch
import torch.nn as nn

from models.ghostnetv3 import GhostModule

class LayerNorm(nn.Module):
    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.std(x, dim=[2, 3], keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-5) + self.beta

# 基础卷积模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
# DWConv
class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

# 残差块，空间局部信息处理
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x

def pywt_forward(x: torch.Tensor, wavelet: str = 'haar', level: int = 2) -> List[torch.Tensor]:
    """使用pywt进行小波包变换，返回各个频率子带的张量列表"""
    # x_np = x.cpu().numpy()
    wp_coeffs = ptwt.wavedec2(x, wavelet=wavelet, level=level)
    return wp_coeffs

def pywt_inverse(coeffs: List[torch.Tensor], wavelet: str = 'haar') -> torch.Tensor:
    """使用pywt进行逆小波包变换，输入是小波包系数的张量列表"""
    # coeffs_np = [coeff.cpu().numpy() for coeff in coeffs]
    reconstructed = ptwt.waverec2(coeffs, wavelet)
    return reconstructed

class FreqBandBranch(nn.Module):
    def __init__(self, in_channels: int, wavelet: str = 'haar', level: int = 2):
        super(FreqBandBranch, self).__init__()
        self.wavelet = wavelet
        self.level = level

        self.conv_low = ConvBlock(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_high1 = ConvBlock(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_high2 = ConvBlock(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_high3 = ConvBlock(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 小波包变换
        wp_coeffs = pywt_forward(x, wavelet=self.wavelet, level=self.level)
        
        # 对每个频率子带应用卷积操作
        # wp_coeffs是一个列表，其中第一个元素通常是近似系数，其余是细节系数
        wp_coeffs[0] = self.conv_low(wp_coeffs[0])  # 处理低频部分
        for i in range(1, len(wp_coeffs)):
            t1 = self.conv_high1(wp_coeffs[i][0])
            t2 = self.conv_high2(wp_coeffs[i][1])
            t3 = self.conv_high3(wp_coeffs[i][2])
            wp_coeffs[i] = (t1, t2, t3)
        # 逆小波包变换
        # 注意，此处将处理过的系数重新组合用于逆变换
        processed_image = pywt_inverse(wp_coeffs, wavelet=self.wavelet)

        return processed_image

class FourierTransformBranch(nn.Module):
    def __init__(self, in_channels):
        super(FourierTransformBranch, self).__init__()
        self.conv_pre_fft = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 调整以确保输出通道数与输入相同
        self.conv_amp = ConvBlock(in_channels, in_channels, kernel_size=1, padding=0)  # 确保输出通道数与输入相同
        self.conv_phase = ConvBlock(in_channels, in_channels, kernel_size=1, padding=0)  # 确保输出通道数与输入相同
    def forward(self, x):
        # 1x1卷积预处理
        x = self.conv_pre_fft(x)
        # 执行FFT
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        # 计算幅度谱和相位谱
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # 对幅度谱和相位谱应用卷积
        amp = self.conv_amp(amp)
        phase = self.conv_phase(phase)
        
        # 逆FFT
        x_back = torch.fft.ifft2(amp * torch.exp(1j * phase), dim=(-2, -1))
        
        return x_back.real

class FreqBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(FreqBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        # 三个分支
        self.spatial_branch = nn.Sequential(
            ResidualBlock(in_channels)
        )
        self.freq_band_branch = nn.Sequential(
            FreqBandBranch(in_channels),
        )
        self.fourier_branch = nn.Sequential(
            FourierTransformBranch(in_channels),
        )
         # 新增：用于融合分支输出的1x1卷积层
        self.concat_conv = GhostModule(3*in_channels, out_channels, kernel_size=1, stride=1)
        self.out_channels = in_channels

    def forward(self, x):
        spatial_out = self.spatial_branch(x)
        # spatial_out + x
        spatial_out = spatial_out + x

        freq_band_out = self.freq_band_branch(x)  # 实现WPT逻辑
        # freq_band_out * silu_ln_x
        fourier_out = self.fourier_branch(x)  # 实现FFT逻辑，并处理幅度与相位
        
        # 合并分支输出
        combined = torch.cat((spatial_out, freq_band_out, fourier_out), dim=1)
        combined_output = self.concat_conv(combined)  # 使用新增的1x1卷积融合特征
        
        return combined_output