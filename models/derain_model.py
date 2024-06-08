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
            weights_init(self.gen, 'normal')
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

        Pixellevel_L1_Loss = self.criterion_L1(self.fake_target, self.true_target)
        ssim_loss = -self.criterion_ssim(self.true_target, self.fake_target)
        SA_perceptual_loss = self.criterionSPL(self.fake_target, self.true_target)
        generator_loss = Pixellevel_L1_Loss + 0.2*ssim_loss + 0.8*SA_perceptual_loss

        return generator_loss