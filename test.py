from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from models.derain_model import DerainModel
from utils.metrics import calculate_psnr
from utils.config import Config
from data.dataset import DenoisingDataset, RainDataset
from utils.transforms import CropWithResize

def validate(model, dataloader, device):
    """
    验证模型性能。
    """
    model.eval()
    psnr_values = []
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            inputs = [inp.to(device) for inp in inputs]
            outputs = model(inputs)
            psnr = calculate_psnr(outputs, inputs[1])
            psnr_values.append(psnr)
            # 显示
            # original = inputs[0].squeeze().permute(1, 2, 0).cpu().numpy()
            # pred = outputs.squeeze().permute(1, 2, 0).cpu().numpy()
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(original)
            # plt.title("Rainy Image")
            # plt.axis("off")
            # plt.subplot(1, 2, 2)
            # plt.imshow(pred)
            # plt.show()
            
    avg_psnr = sum(psnr_values) / len(psnr_values)
    return avg_psnr

def main():
    config = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 数据预处理
    transforms = []
    # transforms.append(CropWithResize(config.crop_size, False))
    transforms.append(ToTensor())
    # transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transforms = Compose(transforms)
    
    # 加载验证数据集
    # 注意：这里假设验证数据集包含雨天图像及其对应的干净图像
    val_dataset = RainDataset(root_dir=config.test_data_dir, transform=transforms, crop=CropWithResize(config.crop_size))
    val_loader = DataLoader(val_dataset, batch_size=config.tbatch_size, shuffle=False)
    
    # 加载训练好的模型
    model_path = config.checkpoint_dir + "/model_epoch_110.pth"  # 假定这是最佳模型的路径
    model = DerainModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 验证模型
    avg_psnr = validate(model, val_loader, device)
    print(f"Average PSNR on validation set: {avg_psnr:.2f} dB")

if __name__ == "__main__":
    main()