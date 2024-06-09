import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from models.base_model import weights_init
from losses.loss_functions import *
import losses.pytorch_ssim as pytorch_ssim

from models.fmodel import FPNet
class DerainModel(nn.Module):
    def __init__(self, config, train=True):
        super(DerainModel, self).__init__()

        self.config = config
        self.gen = FPNet()
        if train:
            weights_init(self.gen, 'normal')
            # 优化器
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gen.parameters()), lr = config.learning_rate, betas = (0.5, 0.999), weight_decay = config.weight_decay)
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=config.cosine_eta_min)
        
            self.criterion_L1 = L1Loss()
            #criterion_L2 = torch.nn.MSELoss().cuda()
            self.criterion_ssim = pytorch_ssim.SSIM().cuda()
            self.criterionSPL = SA_PerceptualLoss().cuda() 
        
    def forward(self, x):
        self.rainy = x[0]
        self.gt = x[1]
        self.out_1, self.out_1_amp, self.out_1_phase, self.out_2 = self.gen(self.rainy)
        return self.out_2  # 输出为去雨后的图像
    
    def compute_loss(self):
        gt_fft = torch.fft.fft2(self.gt, dim=(-2, -1))
        gt_amp = torch.abs(gt_fft)

        rainy_fft = torch.fft.fft2(self.rainy, dim=(-2, -1))
        rainy_phase = torch.angle(rainy_fft)

        gt_first_stage = torch.fft.ifft2(gt_amp * torch.exp(1j * rainy_phase), dim=(-2, -1)).real

        l1 = self.criterion_L1(self.out_1, gt_first_stage)
        l2 = self.criterion_L1(self.out_2, self.gt)
        label_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
        # pred_fft1 = torch.fft.fft2(preds[0],  dim=(-2, -1))
        pred_fft = torch.fft.fft2(self.out_2, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        # f1 = self.cri_l1(pred_fft1, gt_fft)
        l_fft = self.criterion_L1(pred_fft, label_fft)
        l_amp = self.criterion_L1(self.out_1_amp, gt_amp)

        l_perc = self.criterionSPL(self.out_2, self.gt) + self.criterionSPL(self.out_1, gt_first_stage)

        return l1 * 0.1 + l2 + 0.05 * l_fft + 0.05 * l_amp + l_perc * 0.5