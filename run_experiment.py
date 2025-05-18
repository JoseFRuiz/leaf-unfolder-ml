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
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def setup(self, stage: Optional[str] = None):
        # Get list of leaf names (excluding the straightened versions)
        all_files = os.listdir(self.data_dir)
        self.leafnames = [f.split('.')[0] for f in all_files if f.endswith('.JPG') and not f.endswith('F_desdoblada.jpg')]
        
        # Split into train/val sets (80/20 split)
        np.random.shuffle(self.leafnames)
        split_idx = int(len(self.leafnames) * 0.8)
        self.train_leafnames = self.leafnames[:split_idx]
        self.val_leafnames = self.leafnames[split_idx:]

        if stage == 'fit' or stage is None:
            self.train_dataset = LeafDataset(
                self.data_dir, self.train_leafnames, transform=self.transform
            )
            self.val_dataset = LeafDataset(
                self.data_dir, self.val_leafnames, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
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
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.res = ResidualBlock(out_channels + out_channels, out_channels, time_channels)

    def forward(self, x, skip, t):
        x = self.up(x)
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
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Time embedding
        time_channels = 256
        self.time_embed = TimeEmbedding(time_channels)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(3, 64, 3, padding=1)
        
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

    def forward(self, x, t):
        # x: (B, 3, H, W), t: (B,)
        t = self.time_embed(t)  # (B, time_channels)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsampling
        d1 = self.down1(x, t)
        d2 = self.down2(d1, t)
        d3 = self.down3(d2, t)
        
        # Middle
        x = self.middle(d3, t)
        
        # Upsampling
        x = self.up1(x, d3, t)
        x = self.up2(x, d2, t)
        x = self.up3(x, d1, t)
        
        # Final convolution
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
        """Save example predictions for a batch of images."""
        self.eval()
        with torch.no_grad():
            # Select first image from batch
            folded = folded[0:1]  # Keep batch dimension
            straight = straight[0:1]
            
            # Sample timesteps for visualization
            n_steps = 10  # Number of steps to visualize
            timesteps = self.sample_timesteps(n_steps)
            
            # Start from pure noise
            x_t = torch.randn_like(folded)
            
            # Create figure for this example
            fig, axes = plt.subplots(1, n_steps + 2, figsize=(2 * (n_steps + 2), 2))
            
            # Plot input folded image
            folded_np = folded[0].permute(1, 2, 0).cpu().numpy()
            folded_np = (folded_np * 0.5 + 0.5).clip(0, 1)  # Denormalize
            axes[0].imshow(folded_np)
            axes[0].set_title('Folded')
            axes[0].axis('off')
            
            # Plot target straight image
            straight_np = straight[0].permute(1, 2, 0).cpu().numpy()
            straight_np = (straight_np * 0.5 + 0.5).clip(0, 1)  # Denormalize
            axes[1].imshow(straight_np)
            axes[1].set_title('Target')
            axes[1].axis('off')
            
            # Denoise step by step
            for i, t in enumerate(timesteps):
                t_batch = t.repeat(folded.shape[0])
                x_t = self.denoise_step(x_t, t_batch)
                
                # Plot intermediate result
                img_np = x_t[0].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 0.5 + 0.5).clip(0, 1)  # Denormalize
                axes[i + 2].imshow(img_np)
                axes[i + 2].set_title(f't={t.item()}')
                axes[i + 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.example_dir, f'epoch_{epoch:04d}.png'))
            plt.close()
        self.train()

    def training_step(self, batch, batch_idx):
        folded, straight = batch
        batch_size = folded.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        
        # Get noisy image and noise
        x_t, noise = self.get_noisy_image(straight, t)
        
        # Predict noise
        predicted_noise = self(x_t, t)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Log loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        folded, straight = batch
        batch_size = folded.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        
        # Get noisy image and noise
        x_t, noise = self.get_noisy_image(straight, t)
        
        # Predict noise
        predicted_noise = self(x_t, t)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Log loss
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        """Save example predictions at the end of each epoch if needed."""
        if (self.current_epoch + 1) % self.save_examples_every_n_epochs == 0:
            # Get a batch from the validation dataloader
            val_batch = next(iter(self.trainer.val_dataloaders[0]))
            folded, straight = val_batch
            self.save_example_predictions(folded, straight, self.current_epoch + 1)

def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Initialize data module
    data_dir = os.path.join('data', "fotos hojas bromelias")
    data_module = LeafDataModule(data_dir, batch_size=4)

    # Initialize model with example saving every 5 epochs
    model = DiffusionModel(save_examples_every_n_epochs=5)

    # Setup logging and checkpointing
    logger = TensorBoardLogger("lightning_logs", name="leaf_unfolding")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="leaf-unfolding-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )

    # Initialize trainer
    trainer = Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto"
    )

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main() 