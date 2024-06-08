import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from data.dataset import DenoisingDataset, RainDataset

from models.derain_model import DerainModel
from utils.config import Config
from utils.logger import Logger
from utils.transforms import CropWithResize 

def adjust_learning_rate(config, epoch, optimizer):
        target_epoch = config.epochs - config.lr_decrease_epoch
        remain_epoch = config.epochs - epoch
        if epoch >= config.lr_decrease_epoch:
            lr = config.lr * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for i, inputs in enumerate(dataloader):
        # inputs is a list
        # to device
        inputs = [inp.to(device) for inp in inputs]
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)
        loss = model.compute_loss()  # 在模型内部计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        scheduler.step()
        running_loss += loss.item()
        # adjust_learning_rate(model.config, i, optimizer)
    return running_loss / (i + 1)

def main():
    config = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transforms = []
    transforms.append(CropWithResize(config.crop_size))
    transforms.append(ToTensor())
    # transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transforms = Compose(transforms)
    # 加载数据集
    train_dataset = RainDataset(root_dir=config.train_data_dir, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 模型实例化
    model = DerainModel(config).to(device)

    optimizer = model.optimizer
    scheduler = model.scheduler

    logger = Logger(config.log_dir)
    # 训练循环
    num_epochs = config.epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 这里可以添加验证逻辑，类似于训练循环但模型应设置为eval模式
        # 并且可能需要记录和比较验证损失以进行模型选择
        
        # 日志记录
        
        logger.log("train_loss", train_loss, epoch)
        
        # 模型保存逻辑可以根据需要添加
        if (epoch+1) % config.save_model_interval == 0:
            torch.save(model.state_dict(), f"{config.checkpoint_dir}/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()