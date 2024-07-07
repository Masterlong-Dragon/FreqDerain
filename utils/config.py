import os

class Config:
    def __init__(self):
        # 基本训练参数
        self.batch_size = 1
        self.tbatch_size = 1
        self.epochs = 200
        self.learning_rate = 0.0002 / 6
        self.num_workers = 4  # 数据加载线程数
        self.weight_decay = 0
        self.cosine_eta_min = 1e-8
        self.lr_decrease_epoch = 50
        
        # 模型相关
        self.model_name = "SimpleDerainModel"
        self.checkpoint_dir = "./checkpoints"
        self.resume_training = False  # 是否从检查点恢复训练
        self.checkpoint_file = "latest.pth"  # 断点续训时的模型文件名
        
        # 数据路径
        self.data_root = r"F:\EIData\DerainCycleGAN\Rain100L"
        self.train_data_dir = os.path.join(self.data_root, "train")
        self.val_data_dir = os.path.join(self.data_root, "val")
        self.test_data_dir = os.path.join(self.data_root, "test")
        self.crop_size = 256  # 训练时裁剪的尺寸
        
        # 可视化和日志
        self.log_interval = 100  # 打印训练状态的间隔
        self.save_model_interval = 50  # 保存模型的间隔（以epoch计）
        self.log_dir = "./logs/"
  