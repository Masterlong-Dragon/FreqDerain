import os
import torch
from tensorboardX import SummaryWriter
import torchvision

class Logger:
    def __init__(self, log_dir, use_tensorboard=True):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def log(self, scalar_name, value, global_step):
        """Log a scalar value to both console and TensorBoard."""
        print(f"{scalar_name}: {value} at step {global_step}")
        if self.use_tensorboard:
            self.writer.add_scalar(scalar_name, value, global_step)
    
    def log_images(self, tag, images, global_step, n_row=8):
        """Log a batch of images to TensorBoard."""
        images = torch.clamp(images, 0, 1)
        grid = torchvision.utils.make_grid(images, nrow=n_row)
        if self.use_tensorboard:
            self.writer.add_image(tag, grid, global_step)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.use_tensorboard:
            self.writer.close()
        