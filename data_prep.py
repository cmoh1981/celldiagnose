"""
CellDiagnose-AI: Data Preparation & Dataset Classes
====================================================
Scripts for preparing training data and PyTorch Dataset implementations.
"""

import os
import shutil
import random
import json
import zipfile
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# =============================================================================
# Directory Configuration
# =============================================================================

class DataConfig:
    """Centralized data directory configuration."""

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = Path(__file__).parent

        self.project_root = Path(project_root)
        self.data_root = self.project_root / "data"

        # Raw data
        self.raw_downloads = self.data_root / "raw_downloads"

        # Processed data
        self.processed = self.data_root / "processed"

        # Classification paths
        self.classification_root = self.processed / "classification"
        self.classification_train = self.classification_root / "train"
        self.classification_val = self.classification_root / "val"

        # Segmentation paths
        self.segmentation_root = self.processed / "segmentation"
        self.segmentation_train = self.segmentation_root / "train"
        self.segmentation_val = self.segmentation_root / "val"

        # Cell types for classification (LIVECell dataset)
        self.cell_types = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']

    def create_directories(self):
        """Create all required directories."""
        # Classification directories
        for split in ['train', 'val']:
            for cell_type in self.cell_types:
                path = self.classification_root / split / cell_type
                path.mkdir(parents=True, exist_ok=True)

        # Segmentation directories
        for split in ['train', 'val']:
            (self.segmentation_root / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.segmentation_root / split / 'masks').mkdir(parents=True, exist_ok=True)

        # Raw downloads
        self.raw_downloads.mkdir(parents=True, exist_ok=True)

        print(f"[+] Created directory structure at: {self.data_root}")

    def print_structure(self):
        """Print the expected directory structure."""
        structure = """
        data/
        ├── raw_downloads/          <- Drop your zips here
        │
        └── processed/
            │
            ├── classification/     <- For Cell Type/Health classification
            │   ├── train/
            │   │   ├── HEK293/
            │   │   ├── CHO/
            │   │   ├── HeLa/
            │   │   └── MDCK/
            │   └── val/
            │       ├── HEK293/
            │       ├── CHO/
            │       ├── HeLa/
            │       └── MDCK/
            │
            └── segmentation/       <- For Confluency/U-Net training
                ├── train/
                │   ├── images/
                │   └── masks/
                └── val/
                    ├── images/
                    └── masks/
        """
        print(structure)


# =============================================================================
# Data Augmentation Transforms
# =============================================================================

class CellAugmentation:
    """Augmentation transforms optimized for cell microscopy images."""

    @staticmethod
    def get_classification_transforms(
        train: bool = True,
        image_size: int = 224
    ) -> transforms.Compose:
        """
        Get transforms for classification task.

        Args:
            train: If True, include augmentation; if False, only resize/normalize
            image_size: Target image size

        Returns:
            torchvision transforms composition
        """
        # ImageNet normalization (standard for pretrained models)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if train:
            return transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.05
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ])

    @staticmethod
    def get_segmentation_transforms(
        train: bool = True,
        image_size: int = 256
    ) -> Dict[str, Callable]:
        """
        Get transforms for segmentation task.
        Returns separate but synchronized transforms for image and mask.

        Args:
            train: If True, include augmentation
            image_size: Target image size

        Returns:
            Dict with 'image' and 'mask' transform callables
        """

        class SegmentationTransform:
            """Synchronized transforms for image-mask pairs."""

            def __init__(self, train: bool, size: int):
                self.train = train
                self.size = size
                self.normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

            def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                Apply synchronized transforms to image and mask.

                Args:
                    image: RGB numpy array (H, W, 3)
                    mask: Binary mask numpy array (H, W)

                Returns:
                    Tuple of (image_tensor, mask_tensor)
                """
                # Resize
                image = cv2.resize(image, (self.size, self.size))
                mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

                if self.train:
                    # Random horizontal flip
                    if random.random() > 0.5:
                        image = np.fliplr(image).copy()
                        mask = np.fliplr(mask).copy()

                    # Random vertical flip
                    if random.random() > 0.5:
                        image = np.flipud(image).copy()
                        mask = np.flipud(mask).copy()

                    # Random rotation (90 degree increments)
                    k = random.randint(0, 3)
                    if k > 0:
                        image = np.rot90(image, k).copy()
                        mask = np.rot90(mask, k).copy()

                    # Random brightness/contrast for image only
                    if random.random() > 0.5:
                        alpha = 1.0 + random.uniform(-0.1, 0.1)  # contrast
                        beta = random.uniform(-10, 10)           # brightness
                        image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

                # Convert to tensors
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                image_tensor = self.normalize(image_tensor)

                mask_tensor = torch.from_numpy(mask).float() / 255.0
                mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dim

                return image_tensor, mask_tensor

        return SegmentationTransform(train, image_size)


# =============================================================================
# PyTorch Dataset Classes
# =============================================================================

class CellClassificationDataset(Dataset):
    """
    PyTorch Dataset for cell type classification.
    Expects ImageFolder-style directory structure.
    """

    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose = None,
        class_mapping: Dict[str, int] = None
    ):
        """
        Initialize classification dataset.

        Args:
            root_dir: Path to train/ or val/ directory
            transform: Image transforms
            class_mapping: Optional custom class->index mapping
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Discover classes from subdirectories
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        # Create class mapping
        if class_mapping:
            self.class_to_idx = class_mapping
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Collect all image paths and labels
        self.samples = []
        self._load_samples()

        print(f"[+] Loaded {len(self.samples)} images from {len(self.classes)} classes")

    def _load_samples(self):
        """Scan directories and collect image paths."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_counts = defaultdict(int)
        for _, label in self.samples:
            class_counts[label] += 1

        total = len(self.samples)
        weights = []
        for idx in range(len(self.classes)):
            count = class_counts[idx]
            weight = total / (len(self.classes) * count) if count > 0 else 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label].item() for _, label in self.samples]
        return torch.tensor(sample_weights, dtype=torch.float32)


class CellSegmentationDataset(Dataset):
    """
    PyTorch Dataset for cell segmentation (U-Net training).
    Expects paired images and masks in separate directories.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Callable = None
    ):
        """
        Initialize segmentation dataset.

        Args:
            root_dir: Path to train/ or val/ directory (contains images/ and masks/)
            transform: Synchronized transform for image-mask pairs
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'masks'
        self.transform = transform

        # Validate directories exist
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")

        # Collect paired samples
        self.samples = []
        self._load_samples()

        print(f"[+] Loaded {len(self.samples)} image-mask pairs")

    def _load_samples(self):
        """Find matching image-mask pairs."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

        # Get all image files
        image_files = {
            f.stem: f for f in self.images_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        }

        # Find matching masks
        for mask_path in self.masks_dir.iterdir():
            if mask_path.suffix.lower() in valid_extensions:
                stem = mask_path.stem

                # Handle common naming conventions
                # mask might be "image_mask.png" for "image.png"
                clean_stem = stem.replace('_mask', '').replace('_seg', '')

                if clean_stem in image_files:
                    self.samples.append((
                        str(image_files[clean_stem]),
                        str(mask_path)
                    ))
                elif stem in image_files:
                    self.samples.append((
                        str(image_files[stem]),
                        str(mask_path)
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load image (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8) * 255

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Default conversion to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return image, mask


# =============================================================================
# Data Preparation Scripts
# =============================================================================

class LIVECellProcessor:
    """
    Processor for the LIVECell dataset.
    https://sartorius-research.github.io/LIVECell/
    """

    # LIVECell cell type mapping
    CELL_TYPES = {
        'A172': 'Other',
        'BT474': 'Other',
        'BV2': 'Other',
        'Huh7': 'Other',
        'MCF7': 'Other',
        'SHSY5Y': 'Other',
        'SkBr3': 'Other',
        'SKOV3': 'Other',
    }

    def __init__(self, config: DataConfig):
        self.config = config

    def extract_and_process(
        self,
        zip_path: str,
        val_split: float = 0.2,
        seed: int = 42
    ):
        """
        Extract LIVECell zip and organize for training.

        Args:
            zip_path: Path to LIVECell_dataset.zip
            val_split: Fraction for validation
            seed: Random seed for reproducibility
        """
        print(f"[*] Processing LIVECell dataset from: {zip_path}")

        # Extract zip
        extract_dir = self.config.raw_downloads / "livecell_extracted"
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
            print(f"[+] Extracted to: {extract_dir}")

        # Find images and annotations
        # LIVECell structure varies, adapt as needed
        images_found = list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.png"))
        print(f"[+] Found {len(images_found)} images")

        # For segmentation, find annotation files
        # This is a placeholder - adapt to actual LIVECell structure
        print("[!] Note: Customize this processor for your specific LIVECell version")


class DataSplitter:
    """Utility for splitting data into train/val sets."""

    @staticmethod
    def split_classification_data(
        source_dir: str,
        config: DataConfig,
        val_split: float = 0.2,
        seed: int = 42
    ):
        """
        Split a flat directory of images into train/val by class.

        Args:
            source_dir: Directory with class subdirectories
            config: DataConfig instance
            val_split: Fraction for validation
            seed: Random seed
        """
        random.seed(seed)
        source = Path(source_dir)

        for class_dir in source.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue

            class_name = class_dir.name
            if class_name not in config.cell_types:
                print(f"[!] Skipping unknown class: {class_name}")
                continue

            # Get all images
            images = list(class_dir.glob("*.jpg")) + \
                    list(class_dir.glob("*.png")) + \
                    list(class_dir.glob("*.tif"))

            # Split
            random.shuffle(images)
            split_idx = int(len(images) * (1 - val_split))
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            # Copy to destinations
            train_dest = config.classification_train / class_name
            val_dest = config.classification_val / class_name

            for img in train_images:
                shutil.copy2(img, train_dest / img.name)
            for img in val_images:
                shutil.copy2(img, val_dest / img.name)

            print(f"[+] {class_name}: {len(train_images)} train, {len(val_images)} val")

    @staticmethod
    def split_segmentation_data(
        images_dir: str,
        masks_dir: str,
        config: DataConfig,
        val_split: float = 0.2,
        seed: int = 42
    ):
        """
        Split paired image-mask data into train/val.

        Args:
            images_dir: Directory with source images
            masks_dir: Directory with corresponding masks
            config: DataConfig instance
            val_split: Fraction for validation
            seed: Random seed
        """
        random.seed(seed)
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)

        # Find matching pairs
        pairs = []
        for img in images_path.iterdir():
            if img.suffix.lower() in {'.jpg', '.png', '.tif', '.tiff'}:
                # Look for matching mask
                mask_candidates = [
                    masks_path / img.name,
                    masks_path / f"{img.stem}_mask{img.suffix}",
                    masks_path / f"{img.stem}_seg{img.suffix}",
                ]
                for mask in mask_candidates:
                    if mask.exists():
                        pairs.append((img, mask))
                        break

        print(f"[+] Found {len(pairs)} image-mask pairs")

        # Split
        random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - val_split))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        # Copy to destinations
        for img, mask in train_pairs:
            shutil.copy2(img, config.segmentation_train / 'images' / img.name)
            shutil.copy2(mask, config.segmentation_train / 'masks' / mask.name)

        for img, mask in val_pairs:
            shutil.copy2(img, config.segmentation_val / 'images' / img.name)
            shutil.copy2(mask, config.segmentation_val / 'masks' / mask.name)

        print(f"[+] Split: {len(train_pairs)} train, {len(val_pairs)} val")


# =============================================================================
# DataLoader Factory
# =============================================================================

class DataLoaderFactory:
    """Factory for creating PyTorch DataLoaders with proper configuration."""

    @staticmethod
    def get_classification_loaders(
        config: DataConfig,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        use_weighted_sampling: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders for classification.

        Args:
            config: DataConfig instance
            batch_size: Batch size
            num_workers: Number of data loading workers
            image_size: Target image size
            use_weighted_sampling: Balance classes with weighted sampler

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Transforms
        train_transform = CellAugmentation.get_classification_transforms(
            train=True, image_size=image_size
        )
        val_transform = CellAugmentation.get_classification_transforms(
            train=False, image_size=image_size
        )

        # Datasets
        train_dataset = CellClassificationDataset(
            root_dir=str(config.classification_train),
            transform=train_transform
        )
        val_dataset = CellClassificationDataset(
            root_dir=str(config.classification_val),
            transform=val_transform,
            class_mapping=train_dataset.class_to_idx  # Ensure consistent mapping
        )

        # Sampler for class imbalance
        train_sampler = None
        shuffle = True
        if use_weighted_sampling and len(train_dataset) > 0:
            sample_weights = train_dataset.get_sample_weights()
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            shuffle = False  # Can't use both sampler and shuffle

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    @staticmethod
    def get_segmentation_loaders(
        config: DataConfig,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 256
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders for segmentation.

        Args:
            config: DataConfig instance
            batch_size: Batch size
            num_workers: Number of data loading workers
            image_size: Target image size

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Transforms
        train_transform = CellAugmentation.get_segmentation_transforms(
            train=True, image_size=image_size
        )
        val_transform = CellAugmentation.get_segmentation_transforms(
            train=False, image_size=image_size
        )

        # Datasets
        train_dataset = CellSegmentationDataset(
            root_dir=str(config.segmentation_train),
            transform=train_transform
        )
        val_dataset = CellSegmentationDataset(
            root_dir=str(config.segmentation_val),
            transform=val_transform
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for data preparation."""
    import argparse

    parser = argparse.ArgumentParser(description="CellDiagnose-AI Data Preparation")
    parser.add_argument('command', choices=['init', 'split', 'info'],
                       help="Command to run")
    parser.add_argument('--source', type=str, help="Source directory for split command")
    parser.add_argument('--val-split', type=float, default=0.2,
                       help="Validation split fraction (default: 0.2)")

    args = parser.parse_args()

    config = DataConfig()

    if args.command == 'init':
        print("[*] Initializing data directory structure...")
        config.create_directories()
        config.print_structure()

    elif args.command == 'split':
        if not args.source:
            print("[!] Error: --source required for split command")
            return
        print(f"[*] Splitting data from: {args.source}")
        DataSplitter.split_classification_data(args.source, config, args.val_split)

    elif args.command == 'info':
        config.print_structure()

        # Count existing data
        print("\n[*] Current data counts:")
        for split in ['train', 'val']:
            print(f"\n  {split.upper()}:")
            class_dir = config.classification_root / split
            if class_dir.exists():
                for cell_type in config.cell_types:
                    count = len(list((class_dir / cell_type).glob("*")))
                    print(f"    {cell_type}: {count} images")


if __name__ == "__main__":
    main()
