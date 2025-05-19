import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import argparse
from datetime import datetime

class LeafDataset(Dataset):
    def __init__(self, rootdir: str, leafnames: list, output_size_folded: Tuple[int, int] = (128, 128),
                 output_size_straight: Tuple[int, int] = (256, 256), padding: int = 10, transform=None):
        self.rootdir = rootdir
        self.leafnames = leafnames
        self.output_size_folded = output_size_folded
        self.output_size_straight = output_size_straight
        self.padding = padding
        self.transform = transform

    def __len__(self):
        return len(self.leafnames)

    def __getitem__(self, idx):
        leafname = self.leafnames[idx]
        folded_path = os.path.join(self.rootdir, f"{leafname}.JPG")
        straight_path = os.path.join(self.rootdir, f"{leafname}F_desdoblada.jpg")

        # Print paths for debugging
        print(f"Loading images from:\n{folded_path}\n{straight_path}")

        folded = cv2.imread(folded_path)
        straight = cv2.imread(straight_path)

        if folded is None or straight is None:
            raise ValueError(f"Error loading images: {folded_path} or {straight_path}")

        # Apply cropping and processing
        cropped_folded = self.refine_crop(folded, padding=self.padding)
        cropped_straight = self.refine_crop(straight, padding=self.padding)

        squared_folded = self.pad_to_square(cropped_folded)
        squared_straight = self.pad_to_square(cropped_straight)

        resized_folded = cv2.resize(squared_folded, self.output_size_folded)
        resized_straight = cv2.resize(squared_straight, self.output_size_straight)

        # Random rotation for data augmentation
        random_angle = np.random.randint(0, 360)
        h, w = resized_folded.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)
        rotated_folded = cv2.warpAffine(resized_folded, rotation_matrix, (w, h), 
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Convert to RGB
        rotated_folded = cv2.cvtColor(rotated_folded, cv2.COLOR_BGR2RGB)
        resized_straight = cv2.cvtColor(resized_straight, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            folded_tensor = self.transform(rotated_folded)
            straight_tensor = self.transform(resized_straight)
        else:
            folded_tensor = torch.tensor(rotated_folded, dtype=torch.float32).permute(2, 0, 1) / 255.0
            straight_tensor = torch.tensor(resized_straight, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return folded_tensor, straight_tensor

    @staticmethod
    def refine_crop(image, padding=10):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([20, 30, 30])
        upper_bound = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            return image[y:y+h, x:x+w]
        return image

    @staticmethod
    def pad_to_square(image, background_color=(255, 255, 255)):
        h, w, _ = image.shape
        size = max(h, w)
        padded_img = np.full((size, size, 3), background_color, dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        padded_img[y_offset:y_offset+h, x_offset:x_offset+w] = image
        return padded_img

class LeafDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, output_size_folded: Tuple[int, int] = (128, 128),
                 output_size_straight: Tuple[int, int] = (256, 256), padding: int = 10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_size_folded = output_size_folded
        self.output_size_straight = output_size_straight
        self.padding = padding
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def setup(self, stage: Optional[str] = None):
        # Get list of leaf names (excluding the straightened versions)
        all_files = os.listdir(self.data_dir)
        # Use the same pattern as in visualization.py
        self.leafnames = ['Brom01', 'Brom02', 'Brom03', 'Brom04', 'Brom05', 'Brom06']
        
        # Print found leafnames for debugging
        print(f"Using leaf names: {self.leafnames}")
        
        # Split into train/val sets (80/20 split)
        np.random.shuffle(self.leafnames)
        split_idx = int(len(self.leafnames) * 0.8)
        self.train_leafnames = self.leafnames[:split_idx]
        self.val_leafnames = self.leafnames[split_idx:]

        if stage == 'fit' or stage is None:
            self.train_dataset = LeafDataset(
                self.data_dir, self.train_leafnames, output_size_folded=self.output_size_folded,
                output_size_straight=self.output_size_straight, padding=self.padding, transform=self.transform
            )
            self.val_dataset = LeafDataset(
                self.data_dir, self.val_leafnames, output_size_folded=self.output_size_folded,
                output_size_straight=self.output_size_straight, padding=self.padding, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True  # Add this to address the warning
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True  # Add this to address the warning
        )

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.layers = nn.Sequential(
            nn.Linear(1, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, n_channels),
        )

    def forward(self, t):
        # t: (B,)
        t = t.unsqueeze(-1).float()  # (B, 1)
        return self.layers(t)  # (B, n_channels)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_channels, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        # x: (B, C, H, W), t: (B, time_channels)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.down(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        # Adjust the output channels of upsampling to match skip connection
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        # Add a 1x1 convolution to adjust skip connection channels
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        # Input channels is now out_channels (from up) + out_channels (from skip)
        self.res = ResidualBlock(out_channels * 2, out_channels, time_channels)

    def forward(self, x, skip, t):
        x = self.up(x)
        # Ensure skip connection has the same spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        # Adjust skip connection channels using 1x1 convolution
        skip = self.skip_conv(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t)
        return x

class DiffusionModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, n_steps=1000, beta_start=1e-4, beta_end=0.02, save_examples_every_n_epochs=5):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.save_examples_every_n_epochs = save_examples_every_n_epochs
        
        # Define beta schedule
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, n_steps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        
        # Time embedding
        time_channels = 256
        self.time_embed = TimeEmbedding(time_channels)
        
        # Initial convolution: now takes 6 channels (noisy + folded)
        self.init_conv = nn.Conv2d(6, 64, 3, padding=1)
        
        # Downsampling path
        self.down1 = DownBlock(64, 128, time_channels)
        self.down2 = DownBlock(128, 256, time_channels)
        self.down3 = DownBlock(256, 512, time_channels)
        
        # Middle
        self.middle = ResidualBlock(512, 512, time_channels)
        
        # Upsampling path
        self.up1 = UpBlock(512, 256, time_channels)
        self.up2 = UpBlock(256, 128, time_channels)
        self.up3 = UpBlock(128, 64, time_channels)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)

        # Create directory for saving examples
        self.example_dir = "example_predictions"
        os.makedirs(self.example_dir, exist_ok=True)

    def forward(self, x_t, t, folded):
        # Concatenate noisy image and folded image along channel dimension
        x = torch.cat([x_t, folded], dim=1)  # (B, 6, H, W)
        t = self.time_embed(t)  # (B, time_channels)
        x = self.init_conv(x)
        d1 = self.down1(x, t)
        d2 = self.down2(d1, t)
        d3 = self.down3(d2, t)
        x = self.middle(d3, t)
        x = self.up1(x, d3, t)
        x = self.up2(x, d2, t)
        x = self.up3(x, d1, t)
        x = self.final_conv(x)
        return x

    def get_noisy_image(self, x_0, t):
        # x_0: (B, 3, H, W), t: (B,)
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def sample_timesteps(self, n_steps):
        """Sample n_steps evenly spaced timesteps for visualization."""
        return torch.linspace(0, self.n_steps - 1, n_steps, device=self.device).long()

    def denoise_step(self, x_t, t):
        """Single denoising step."""
        predicted_noise = self(x_t, t)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        
        if t[0] > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
            
        x_t_minus_1 = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
        return x_t_minus_1

    def save_example_predictions(self, folded, straight, epoch):
        self.eval()
        with torch.no_grad():
            folded = folded[0:1].to(self.device)
            straight = straight[0:1].to(self.device)
            n_steps = 10
            timesteps = self.sample_timesteps(n_steps)
            x_t = torch.randn_like(straight)
            # Ensure folded and x_t have the same spatial size
            if folded.shape[2:] != x_t.shape[2:]:
                folded = F.interpolate(folded, size=x_t.shape[2:], mode='bilinear', align_corners=False)
            fig, axes = plt.subplots(1, n_steps + 2, figsize=(2 * (n_steps + 2), 2))
            folded_np = folded[0].permute(1, 2, 0).cpu().numpy()
            folded_np = (folded_np * 0.5 + 0.5).clip(0, 1)
            axes[0].imshow(folded_np)
            axes[0].set_title('Input (Folded)')
            axes[0].axis('off')
            straight_np = straight[0].permute(1, 2, 0).cpu().numpy()
            straight_np = (straight_np * 0.5 + 0.5).clip(0, 1)
            axes[1].imshow(straight_np)
            axes[1].set_title('Target (Straight)')
            axes[1].axis('off')
            for i, t in enumerate(timesteps):
                t_batch = t.repeat(folded.shape[0])
                predicted_noise = self(x_t, t_batch, folded)
                alpha_t = self.alpha[t].view(-1, 1, 1, 1)
                alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                noise = torch.randn_like(x_t) if t.item() > 0 else torch.zeros_like(x_t)
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
                img_np = x_t[0].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 0.5 + 0.5).clip(0, 1)
                progress = 100 * (1 - t.item() / (self.n_steps - 1))
                axes[i + 2].imshow(img_np)
                axes[i + 2].set_title(f'Step {i+1}/{n_steps}\n({progress:.0f}% denoised)')
                axes[i + 2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.example_dir, f'epoch_{epoch:04d}.png'))
            plt.close()

            # Additional comparison plot: input, predicted, target
            predicted_unfolded = x_t[0].permute(1, 2, 0).cpu().numpy()
            predicted_unfolded = (predicted_unfolded * 0.5 + 0.5).clip(0, 1)
            folded_np = folded[0].permute(1, 2, 0).cpu().numpy()
            folded_np = (folded_np * 0.5 + 0.5).clip(0, 1)
            straight_np = straight[0].permute(1, 2, 0).cpu().numpy()
            straight_np = (straight_np * 0.5 + 0.5).clip(0, 1)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(folded_np)
            plt.title('Input (Folded)')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(predicted_unfolded)
            plt.title('Predicted (Unfolded)')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(straight_np)
            plt.title('Target (Straight)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.example_dir, f'comparison_epoch_{epoch:04d}.png'))
            plt.close()
        self.train()

    def training_step(self, batch, batch_idx):
        folded, straight = batch
        batch_size = folded.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        x_t, noise = self.get_noisy_image(straight, t)
        # Ensure folded and x_t have the same spatial size
        if folded.shape[2:] != x_t.shape[2:]:
            folded = F.interpolate(folded, size=x_t.shape[2:], mode='bilinear', align_corners=False)
        predicted_noise = self(x_t, t, folded)
        loss = F.mse_loss(predicted_noise, noise)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        folded, straight = batch
        batch_size = folded.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        x_t, noise = self.get_noisy_image(straight, t)
        # Ensure folded and x_t have the same spatial size
        if folded.shape[2:] != x_t.shape[2:]:
            folded = F.interpolate(folded, size=x_t.shape[2:], mode='bilinear', align_corners=False)
        predicted_noise = self(x_t, t, folded)
        loss = F.mse_loss(predicted_noise, noise)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        """Save example predictions at the end of each epoch if needed."""
        if (self.current_epoch + 1) % self.save_examples_every_n_epochs == 0:
            # Get a batch from the validation dataloader
            val_dataloader = self.trainer.val_dataloaders
            if val_dataloader:
                val_batch = next(iter(val_dataloader))
                folded, straight = val_batch
                self.save_example_predictions(folded, straight, self.current_epoch + 1)

def parse_args():
    parser = argparse.ArgumentParser(description='Train leaf unfolding diffusion model')
    
    # Model hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate for the model')
    parser.add_argument('--n_steps', type=int, default=1000,
                      help='Number of diffusion steps')
    parser.add_argument('--beta_start', type=float, default=1e-4,
                      help='Starting beta value for noise schedule')
    parser.add_argument('--beta_end', type=float, default=0.02,
                      help='Ending beta value for noise schedule')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100,
                      help='Maximum number of training epochs')
    parser.add_argument('--save_examples_every_n_epochs', type=int, default=5,
                      help='Save example predictions every N epochs')
    
    # Data hyperparameters
    parser.add_argument('--folded_size', type=int, default=128,
                      help='Size of folded leaf images')
    parser.add_argument('--straight_size', type=int, default=256,
                      help='Size of straightened leaf images')
    parser.add_argument('--padding', type=int, default=300,
                      help='Padding for leaf cropping')
    
    return parser.parse_args()

def get_experiment_name(args):
    """Generate experiment name based on hyperparameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [
        f"lr{args.learning_rate:.0e}",
        f"steps{args.n_steps}",
        f"bs{args.batch_size}",
        f"folded{args.folded_size}",
        f"straight{args.straight_size}"
    ]
    return f"{timestamp}_{'_'.join(name_parts)}"

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Generate experiment name
    experiment_name = get_experiment_name(args)
    
    # Create experiment directories
    example_dir = os.path.join("example_predictions", experiment_name)
    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(example_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize data module
    data_dir = os.path.join('data', "fotos hojas bromelias")
    data_module = LeafDataModule(
        data_dir, 
        batch_size=args.batch_size,
        output_size_folded=(args.folded_size, args.folded_size),
        output_size_straight=(args.straight_size, args.straight_size),
        padding=args.padding
    )

    # Initialize model with hyperparameters
    model = DiffusionModel(
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        save_examples_every_n_epochs=args.save_examples_every_n_epochs
    )
    model.example_dir = example_dir  # Update example directory

    # Setup logging and checkpointing
    logger = TensorBoardLogger("lightning_logs", name=experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="leaf-unfolding-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    # Initialize trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto"
    )

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main() 