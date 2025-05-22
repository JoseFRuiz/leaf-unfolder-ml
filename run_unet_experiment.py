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
    def __init__(self, rootdir: str, leafnames: list, output_size: Tuple[int, int] = (256, 256), padding: int = 10, transform=None):
        self.rootdir = rootdir
        self.leafnames = leafnames
        self.output_size = output_size
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
        cropped_folded = self.refine_crop(folded, padding=self.padding)
        cropped_straight = self.refine_crop(straight, padding=self.padding)
        squared_folded = self.pad_to_square(cropped_folded)
        squared_straight = self.pad_to_square(cropped_straight)
        resized_folded = cv2.resize(squared_folded, self.output_size)
        resized_straight = cv2.resize(squared_straight, self.output_size)
        # Convert to RGB
        folded_rgb = cv2.cvtColor(resized_folded, cv2.COLOR_BGR2RGB)
        straight_rgb = cv2.cvtColor(resized_straight, cv2.COLOR_BGR2RGB)
        # Apply transforms
        if self.transform:
            folded_tensor = self.transform(folded_rgb)
            straight_tensor = self.transform(straight_rgb)
        else:
            folded_tensor = torch.tensor(folded_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            straight_tensor = torch.tensor(straight_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
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
    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 4, output_size: Tuple[int, int] = (256, 256), padding: int = 10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_size = output_size
        self.padding = padding
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def setup(self, stage: Optional[str] = None):
        all_files = os.listdir(self.data_dir)
        self.leafnames = ['Brom01', 'Brom02', 'Brom03', 'Brom04', 'Brom05', 'Brom06']
        np.random.shuffle(self.leafnames)
        split_idx = int(len(self.leafnames) * 0.8)
        self.train_leafnames = self.leafnames[:split_idx]
        self.val_leafnames = self.leafnames[split_idx:]
        if stage == 'fit' or stage is None:
            self.train_dataset = LeafDataset(
                self.data_dir, self.train_leafnames, output_size=self.output_size, padding=self.padding, transform=self.transform
            )
            self.val_dataset = LeafDataset(
                self.data_dir, self.val_leafnames, output_size=self.output_size, padding=self.padding, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class SimpleUNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out

    def training_step(self, batch, batch_idx):
        folded, straight = batch
        pred = self(folded)
        loss = F.mse_loss(pred, straight)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        folded, straight = batch
        pred = self(folded)
        loss = F.mse_loss(pred, straight)
        self.log('val_loss', loss, prog_bar=True)
        if batch_idx == 0:
            self.save_example_predictions(folded, pred, straight, self.current_epoch)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def save_example_predictions(self, folded, pred, straight, epoch):
        folded = folded[0].detach().cpu().permute(1, 2, 0).numpy()
        pred = pred[0].detach().cpu().permute(1, 2, 0).numpy()
        straight = straight[0].detach().cpu().permute(1, 2, 0).numpy()
        folded = (folded * 0.5 + 0.5).clip(0, 1)
        pred = (pred * 0.5 + 0.5).clip(0, 1)
        straight = (straight * 0.5 + 0.5).clip(0, 1)
        os.makedirs('unet_example_predictions', exist_ok=True)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(folded)
        plt.title('Input (Folded)')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(pred)
        plt.title('Predicted (Straight)')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(straight)
        plt.title('Target (Straight)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'unet_example_predictions/epoch_{epoch:04d}.png')
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net for leaf unfolding')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--padding', type=int, default=300, help='Padding for cropping')
    return parser.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(42)
    data_dir = os.path.join('data', "fotos hojas bromelias")
    data_module = LeafDataModule(
        data_dir,
        batch_size=args.batch_size,
        output_size=(args.img_size, args.img_size),
        padding=args.padding,
        num_workers=0  # Set to 0 for debugging
    )
    model = SimpleUNet(learning_rate=args.learning_rate)
    logger = TensorBoardLogger("unet_logs", name="leaf_unfolding_unet")
    checkpoint_callback = ModelCheckpoint(
        dirpath="unet_checkpoints",
        filename="unet-leaf-unfolding-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto"
    )
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main() 