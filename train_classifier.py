#!/usr/bin/env python3
"""
CellDiagnose-AI: Classification Model Training
===============================================
Train EfficientNet-B3 for cell type classification with class imbalance handling.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import numpy as np
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_classifier.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class CellDataset(Dataset):
    """Cell classification dataset with augmentation."""

    def __init__(self, root_dir: str, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Collect all images
        self.samples = []
        self.class_names = []

        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                self.class_names.append(class_dir.name)
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, class_dir.name))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((img_path, class_dir.name))

        # Create class mapping
        if class_to_idx is None:
            self.class_to_idx = {name: idx for idx, name in enumerate(sorted(set(self.class_names)))}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        # Get class counts for weighted sampling
        self.class_counts = Counter([s[1] for s in self.samples])

        logger.info(f"Loaded {len(self.samples)} images from {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]

        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        label = self.class_to_idx[class_name]

        return image, label

    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler."""
        class_weights = {cls: 1.0 / count for cls, count in self.class_counts.items()}
        weights = [class_weights[s[1]] for s in self.samples]
        return weights


# =============================================================================
# Transforms
# =============================================================================

def get_train_transforms(img_size=300):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
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


def get_val_transforms(img_size=300):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# =============================================================================
# Model
# =============================================================================

class CellClassifier(nn.Module):
    """EfficientNet-B3 based cell classifier."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B3
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features  # 1536 for B3

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """Freeze backbone for initial training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# =============================================================================
# Training
# =============================================================================

class Trainer:
    """Training manager for cell classifier."""

    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Class weights for focal loss
        class_counts = train_loader.dataset.class_counts
        class_to_idx = train_loader.dataset.class_to_idx
        total = sum(class_counts.values())
        num_classes = len(class_to_idx)

        # Compute inverse frequency weights (ordered by class index)
        weights = []
        for cls in sorted(class_to_idx.keys()):
            count = class_counts.get(cls, 1)  # Default to 1 if class not in training
            weights.append(total / (num_classes * count))
        self.class_weights = torch.FloatTensor(weights).to(device)

        # Loss function
        self.criterion = FocalLoss(gamma=2.0, weight=self.class_weights)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )

        # Mixed precision
        self.scaler = GradScaler('cuda')

        # Tracking
        self.best_val_acc = 0
        self.best_val_balanced_acc = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_balanced_acc': []}

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward
            with autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Stats
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        self.scheduler.step()

        return running_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)

        # Balanced accuracy
        classes = np.unique(all_labels)
        per_class_acc = []
        for cls in classes:
            mask = all_labels == cls
            if mask.sum() > 0:
                per_class_acc.append((all_preds[mask] == cls).sum() / mask.sum())
        balanced_acc = 100. * np.mean(per_class_acc)

        return running_loss / len(self.val_loader), accuracy, balanced_acc

    def train(self, num_epochs, checkpoint_dir='checkpoints'):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Class weights: {self.class_weights.cpu().numpy()}")

        for epoch in range(num_epochs):
            # Phase 1: Frozen backbone (first 10 epochs)
            if epoch == 0:
                logger.info("Phase 1: Training classifier head only")
                self.model.freeze_backbone()
            elif epoch == 10:
                logger.info("Phase 2: Fine-tuning entire model")
                self.model.unfreeze_backbone()
                # Reduce learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['lr'] / 10

            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_balanced_acc = self.validate()

            # Log
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"Val Balanced Acc: {val_balanced_acc:.2f}%"
            )

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_balanced_acc'].append(val_balanced_acc)

            # Save best model
            if val_balanced_acc > self.best_val_balanced_acc:
                self.best_val_balanced_acc = val_balanced_acc
                self.best_val_acc = val_acc

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_balanced_acc': val_balanced_acc,
                    'class_to_idx': self.train_loader.dataset.class_to_idx,
                    'history': self.history,
                }, checkpoint_dir / 'classifier_best.pth')

                logger.info(f"Saved best model (balanced acc: {val_balanced_acc:.2f}%)")

            # Save latest
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history,
            }, checkpoint_dir / 'classifier_latest.pth')

        logger.info(f"Training complete! Best balanced accuracy: {self.best_val_balanced_acc:.2f}%")
        return self.history


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train cell classifier')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=300, help='Image size')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    args = parser.parse_args()

    # Paths
    train_dir = 'data/processed/classification/train'
    val_dir = 'data/processed/classification/val'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Datasets
    train_dataset = CellDataset(train_dir, transform=get_train_transforms(args.img_size))
    val_dataset = CellDataset(
        val_dir,
        transform=get_val_transforms(args.img_size),
        class_to_idx=train_dataset.class_to_idx
    )

    # Weighted sampler for class imbalance
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    logger.info(f"Train: {len(train_dataset)} images, {train_dataset.num_classes} classes")
    logger.info(f"Val: {len(val_dataset)} images")
    logger.info(f"Class distribution: {dict(train_dataset.class_counts)}")

    # Model
    model = CellClassifier(num_classes=train_dataset.num_classes, pretrained=True)
    logger.info(f"Model: EfficientNet-B3 with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Config
    config = {
        'lr': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
    }

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train(args.epochs)

    # Save final results
    with open('training_results.json', 'w') as f:
        json.dump({
            'config': config,
            'best_val_acc': trainer.best_val_acc,
            'best_val_balanced_acc': trainer.best_val_balanced_acc,
            'history': history,
            'class_to_idx': train_dataset.class_to_idx,
        }, f, indent=2)

    logger.info("Training results saved to training_results.json")


if __name__ == '__main__':
    main()
