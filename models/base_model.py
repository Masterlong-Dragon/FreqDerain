import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from models.spectral import SpectralNorm

from models.model_utils import FreqBlock, DownLayer, ConvBlock

from models.ghostnetv3 import GhostModule

class DownBlockWithFreq(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlockWithFreq, self).__init__()
        # 确保额外的卷积层用于调整 FreqBlock 的输出通道数
        self.freq_to_out_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.freq_block = FreqBlock(in_channels)
        self.attn = Self_Attn(out_channels, 'relu')  # 注意力机制应用于空间路径输出前
        self.down = ConvBlock(in_channels, out_channels, stride=2)
        self.down_freq = ConvBlock(in_channels, out_channels, stride=2)
        self.fusion_conv = ConvBlock(2 * out_channels, out_channels)

    def forward(self, spatial_in, freq_in):
        spatial_out = self.down(spatial_in)  # 下采样
        spatial_att = self.attn(spatial_out)  # 应用注意力
        # 频域路径处理，先通过 FreqBlock 再下采样
        freq_out = self.freq_block(freq_in)
        # freq_out_adjusted = self.freq_to_out_channels(freq_out)  # 调整 FreqBlock 输出通道数
        freq_out_down = self.down_freq(freq_out)
        
        # 确保融合前的通道数一致
        fused = torch.cat([spatial_att, freq_out_down], dim=1)
        fused = self.fusion_conv(fused)  # 融合后通过卷积调整通道至 out_channels
        
        # 返回空间路径输出、频域路径输出（用于跳连）及融合后的输出
        return spatial_out, freq_out_down, fused
    
class UpBlockWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpBlockWithSkip, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.freq_block = FreqBlock(out_channels)
        self.freq_to_out_channels = nn.Conv2d(skip_channels + out_channels // 2, out_channels, kernel_size=1) if out_channels // 2 != out_channels else nn.Identity()

    def forward(self, x, skip_conn):
        x = self.up(x)
        x = torch.cat([x, skip_conn], dim=1)
        x = self.freq_to_out_channels(x)
        x = self.freq_block(x)
        return x

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
    
# ----------------------------------------
#      Generator
# ----------------------------------------
class Generator (nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        out_channel = 64
        # down = maxpooling+3*conv
        self.inc = TripleConvs(3, 64)
        self.freq_inc = FreqBlock(64)
        self.conv1 = DownBlockWithFreq(64, 128)
        self.conv2 = DownBlockWithFreq(128, 256)
        self.conv3 = DownBlockWithFreq(256, 512)
        self.conv4 = DownBlockWithFreq(512, 512)
        self.FeatureSA = FreqBlock(512)  
        # up = deconv + 3*conv
        self.conv6 = UpBlockWithSkip(512, 256, 512)
        self.conv7 = UpBlockWithSkip(256, 128, 256)
        self.conv8 = UpBlockWithSkip(128, 64, 128)
        self.outc = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        )
        self.fm = nn.Conv2d(out_channel, 1, 3, 1, 1)  #feature map
        
        #Rain de-raining Module
        self.rd_conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, 1, 2),
            nn.ReLU()
            )
        self.rd_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
            )
        self.rd_conv3 = nn.Sequential(
            # nn.Conv2d(128, 128, 3, 1, 1),
            # nn.ReLU()
            FreqBlock(128)
            )
        self.rd_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
            )
        self.rd_conv5 = nn.Sequential(
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.ReLU()
            FreqBlock(256)
            )
        self.rd_conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.DerainSA = Self_Attn(256, 'relu')  #self-attention
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation = 2),
            nn.ReLU()
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation = 4),
            nn.ReLU()
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation = 8),
            nn.ReLU()
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation = 16),
            nn.ReLU()
            )
        self.rd_conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.rd_conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.rd_conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.rd_conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
            )

    def forward(self, Rainy_image_x):
        #Rain Feature Extraction Module:
        #:param data_with_est: if not blind estimation, it is same as data
        #:param data:
        #:return: img_pred, img_featuremap
        #down-sampling
        Rainy_image = Rainy_image_x
        inc = self.inc(Rainy_image_x)
        freq_inc = self.freq_inc(inc)
        spatial_out1, freq_out1, fused1 = self.conv1(inc, freq_inc)
        spatial_out2, freq_out2, fused2 = self.conv2(spatial_out1, fused1)
        spatial_out3, freq_out3, fused3 = self.conv3(spatial_out2, fused2)
        spatial_out4, freq_out4, fused4 = self.conv4(spatial_out3, fused3)
        feature_sa = self.FeatureSA(fused4)  #self-attention
        #up-sampling + crop
        conv6 = self.conv6(feature_sa, freq_out3)
        conv7 = self.conv7(conv6, freq_out2)
        conv8 = self.conv8(conv7, freq_out1)
        # return channel K*K*N
        feature_map = self.outc(conv8)
        feature_map = self.fm(feature_map)  #feature map
        
        #Rain de-raining Module: skip-network
        x = torch.cat((Rainy_image, feature_map), 1)
        x = self.rd_conv1(x)
        res1 = x
        x = self.rd_conv2(x)  #down
        x = self.rd_conv3(x)
        res2 = x
        x = self.rd_conv4(x)  #down
        x = self.rd_conv5(x)
        x = self.rd_conv6(x)
        x = self.DerainSA(x)  #self-attention
        x = self.diconv1(x)  #dilated conv
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.rd_conv7(x)
        x = self.rd_conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)  #up
        x = x + res2
        x = self.rd_conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)  #up
        x = x + res1
        x = self.rd_conv10(x)
        pred = self.output(x)
        return pred, feature_map

class TripleConvs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.MaxPool2d(2),
            TripleConvs(in_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, channels_x, channels_y, channels_out):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels_y, channels_y, 4, 2, 1)
        self.dconv = TripleConvs(channels_x+channels_y, channels_out)

    def forward(self, x, y):
        deconv_y = self.deconv(y)
        cont_xy = torch.cat([x, deconv_y], dim=1)
        dconvxy = self.dconv(cont_xy)
        return dconvxy

# ----------------------------------------
#      Self-attention (SAGAN ver.)
# ----------------------------------------
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        #return out,attention
        return out