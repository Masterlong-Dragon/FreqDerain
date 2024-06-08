import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.transforms import Normalize

# 常规损失函数
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)

# VGG16感知损失
class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layer=19, device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = vgg16(pretrained=True).features[:feature_layer].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
        
    def forward(self, pred, target):
        pred_features = self.vgg(self.norm(pred))
        target_features = self.vgg(self.norm(target))
        return torch.mean(torch.abs(pred_features - target_features))
    

class SA_PerceptualLoss(nn.Module):
    def __init__(self):
        super(SA_PerceptualLoss, self).__init__()
        self.model = (vgg16(pretrained=True)).cuda()
        self.trainable(self.model, False)
        self.vgg_layers = self.model.features
        self.FeatureSA = Self_Attn(512, 'relu')  #self-attention
        self.layer_name_mapping = {
            '6': "relu2_1",  #128
            '8': "relu2_2",   #128
            '29': "relu5_3"   #512
        }
        self.loss = nn.MSELoss().cuda()

    def trainable(self, net, trainable):
        for param in net.parameters():
            param.requires_grad = trainable

    def vgg_output(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                if name == "relu5_3":
                    x=self.FeatureSA(x)
                output.append(x)
        return output

    def __call__(self, O, T):
        vgg_O = self.vgg_output(O)
        vgg_T = self.vgg_output(T)

        output_len = len(vgg_T)

        sa_perceptual_loss = None
        for i in range(output_len):
            if i == 0:
                sa_perceptual_loss = self.loss(vgg_O[i], vgg_T[i]) / float(output_len)
            else:
                sa_perceptual_loss += self.loss(vgg_O[i], vgg_T[i]) / float(output_len)
        return sa_perceptual_loss

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