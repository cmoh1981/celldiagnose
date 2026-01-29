#!/usr/bin/env python3
"""
CellDiagnose-AI: Segmentation Model Training (PyTorch Lightning)
================================================================
Train U-Net with ResNet-34 encoder for cell instance segmentation.
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_segmentation.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class CellSegmentationDataset(Dataset):
    """Cell segmentation dataset with paired images and masks."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'masks'

        # Collect all image files
        self.image_files = sorted(list(self.images_dir.glob('*.png')))

        logger.info(f"Loaded {len(self.image_files)} images from {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        # Load image and mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Convert instance mask to binary mask (0 = background, 1 = cell)
        mask = (mask > 0).astype(np.float32)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Ensure mask has channel dimension
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


# =============================================================================
# Data Module
# =============================================================================

class CellSegmentationDataModule(L.LightningDataModule):
    """Lightning DataModule for cell segmentation."""

    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int = 8,
        img_size: int = 512,
        num_workers: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

    def get_train_transforms(self):
        return A.Compose([
            A.RandomResizedCrop(
                size=(self.img_size, self.img_size),
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.2),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.CLAHE(clip_limit=2.0),
            ], p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def get_val_transforms(self):
        return A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CellSegmentationDataset(
                self.train_dir,
                transform=self.get_train_transforms()
            )
            self.val_dataset = CellSegmentationDataset(
                self.val_dir,
                transform=self.get_val_transforms()
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# =============================================================================
# Loss Functions
# =============================================================================

class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross Entropy loss."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE loss
        bce_loss = self.bce(logits, targets)

        # Dice loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# =============================================================================
# Lightning Module
# =============================================================================

class CellSegmentationModel(L.LightningModule):
    """U-Net with ResNet-34 encoder for cell segmentation."""

    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        lr: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create U-Net model
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None  # We'll apply sigmoid in loss/metrics
        )

        # Loss function
        self.criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)

        # Metrics
        self.train_dice = []
        self.val_dice = []
        self.val_iou = []

    def forward(self, x):
        return self.model(x)

    def compute_dice(self, logits, targets, threshold=0.5):
        """Compute Dice coefficient."""
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return dice.mean()

    def compute_iou(self, logits, targets, threshold=0.5):
        """Compute IoU (Jaccard) coefficient."""
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        loss = self.criterion(logits, masks)
        dice = self.compute_dice(logits, masks)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_dice', dice, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        loss = self.criterion(logits, masks)
        dice = self.compute_dice(logits, masks)
        iou = self.compute_iou(logits, masks)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_dice', dice, prog_bar=True, sync_dist=True)
        self.log('val_iou', iou, prog_bar=True, sync_dist=True)

        return {'val_loss': loss, 'val_dice': dice, 'val_iou': iou}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train cell segmentation model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=512, help='Image size')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder name')
    args = parser.parse_args()

    # Paths
    train_dir = 'data/processed/segmentation/train'
    val_dir = 'data/processed/segmentation/val'

    logger.info(f"Training U-Net segmentation model")
    logger.info(f"Encoder: {args.encoder}")
    logger.info(f"Image size: {args.img_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")

    # Data module
    data_module = CellSegmentationDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.workers
    )

    # Model
    model = CellSegmentationModel(
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        lr=args.lr
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params/1e6:.2f}M")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='segmentation-{epoch:02d}-{val_dice:.4f}',
        save_top_k=3,
        monitor='val_dice',
        mode='max',
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor='val_dice',
        patience=15,
        mode='max',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Logger
    csv_logger = CSVLogger('logs', name='segmentation')

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=csv_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size = 16
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module)

    # Save best model path
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Best val_dice: {checkpoint_callback.best_model_score:.4f}")

    # Copy best model to standard location
    if best_model_path:
        import shutil
        shutil.copy(best_model_path, 'checkpoints/segmentation_best.pth')
        logger.info("Copied best model to checkpoints/segmentation_best.pth")


if __name__ == '__main__':
    main()
