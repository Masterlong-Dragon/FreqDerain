import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from models.base_model import Generator, weights_init
from losses.loss_functions import *
import losses.pytorch_ssim as pytorch_ssim
class DerainModel(nn.Module):
    def __init__(self, config, train=True):
        super(DerainModel, self).__init__()

        self.config = config
        self.gen = Generator()
        if train:
            weights_init(self.gen, 'xavier')
            # 优化器
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gen.parameters()), lr = config.learning_rate, betas = (0.5, 0.999), weight_decay = config.weight_decay)
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=config.cosine_eta_min)
        
            self.criterion_L1 = L1Loss()
            #criterion_L2 = torch.nn.MSELoss().cuda()
            self.criterion_ssim = pytorch_ssim.SSIM().cuda()
            self.criterionSPL = SA_PerceptualLoss().cuda() 
        
    def forward(self, x):
        self.true_target = x[1]
        self.fake_target, feature_map = self.gen(x[0])
        return self.fake_target  # 输出为去雨后的图像
    
    def compute_loss(self):

        # fft正则项
        gt_fft = torch.fft.fft2(self.true_target, dim=(-2, -1))
        fake_fft = torch.fft.fft2(self.fake_target, dim=(-2, -1))
        # 相位
        gt_phase = torch.angle(gt_fft)
        fake_phase = torch.angle(fake_fft)
        # 幅度
        gt_amp = torch.abs(gt_fft)
        fake_amp = torch.abs(fake_fft)
        # 计算fft正则项
        fft_loss = self.criterion_L1(fake_phase, gt_phase) + self.criterion_L1(fake_amp, gt_amp)

        Pixellevel_L1_Loss = self.criterion_L1(self.fake_target, self.true_target)
        ssim_loss = -self.criterion_ssim(self.true_target, self.fake_target)
        SA_perceptual_loss = self.criterionSPL(self.fake_target, self.true_target)
        generator_loss = Pixellevel_L1_Loss + 0.2*ssim_loss + 0.8*SA_perceptual_loss + 0.005*fft_loss

        return generator_loss