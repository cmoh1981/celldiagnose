#!/usr/bin/env python3
"""
CellDiagnose-AI: Unified Inference Pipeline
============================================
Single API for complete cell microscopy diagnostics.

Usage:
    from celldiagnose import CellDiagnoseAI

    # Initialize with pretrained models
    model = CellDiagnoseAI.load('checkpoints/')

    # Run complete diagnosis
    result = model.diagnose('cell_image.png')

    # Access results
    print(result.cell_type)           # 'HeLa'
    print(result.cell_type_confidence) # 0.987
    print(result.confluency)          # 45.2
    print(result.cell_count)          # 127
    print(result.health_status)       # 'healthy'
    print(result.anomaly_score)       # 0.023

    # Get segmentation mask
    mask = result.segmentation_mask   # numpy array

    # Generate visualization
    vis = result.visualize()          # overlay image

    # Export report
    result.to_json('report.json')
    result.to_dict()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, Tuple
import json
from datetime import datetime
import warnings

# Optional imports
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn("timm not available. Classification will use fallback model.")

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    warnings.warn("segmentation_models_pytorch not available. Segmentation will use fallback.")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ClassificationResult:
    """Cell classification result."""
    cell_type: str
    confidence: float
    all_probabilities: Dict[str, float]
    health_status: str = "unknown"
    health_confidence: float = 0.0


@dataclass
class SegmentationResult:
    """Cell segmentation result."""
    mask: np.ndarray
    confluency_percent: float
    cell_count: int
    overlay: Optional[np.ndarray] = None


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    score: float
    normalized_score: float
    is_anomaly: bool
    severity: str  # normal, low, medium, high, critical
    heatmap: Optional[np.ndarray] = None
    possible_issues: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class DiagnosisResult:
    """Complete diagnosis result."""
    # Image info
    image_shape: Tuple[int, int, int]
    timestamp: str

    # Classification
    cell_type: str
    cell_type_confidence: float
    cell_type_probabilities: Dict[str, float]

    # Segmentation
    confluency: float
    cell_count: int
    segmentation_mask: np.ndarray

    # Health assessment
    health_status: str  # healthy, needs_attention, unhealthy
    anomaly_score: float
    anomaly_severity: str
    possible_issues: List[str]
    recommendation: str

    # Raw results
    _classification: ClassificationResult = field(repr=False, default=None)
    _segmentation: SegmentationResult = field(repr=False, default=None)
    _anomaly: AnomalyResult = field(repr=False, default=None)
    _original_image: np.ndarray = field(repr=False, default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding numpy arrays)."""
        # Convert numpy types to Python types
        def to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_python(v) for v in obj]
            return obj

        return {
            'image_shape': list(self.image_shape),
            'timestamp': self.timestamp,
            'cell_type': self.cell_type,
            'cell_type_confidence': to_python(self.cell_type_confidence),
            'cell_type_probabilities': to_python(self.cell_type_probabilities),
            'confluency': to_python(self.confluency),
            'cell_count': to_python(self.cell_count),
            'health_status': self.health_status,
            'anomaly_score': to_python(self.anomaly_score),
            'anomaly_severity': self.anomaly_severity,
            'possible_issues': self.possible_issues,
            'recommendation': self.recommendation,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Export to JSON string or file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        if path:
            Path(path).write_text(json_str)
        return json_str

    def visualize(self, show_heatmap: bool = True) -> np.ndarray:
        """Create visualization with segmentation overlay and optional anomaly heatmap."""
        if self._original_image is None:
            raise ValueError("Original image not available for visualization")

        image = self._original_image.copy()
        h, w = image.shape[:2]

        # Create segmentation overlay (green)
        mask_resized = cv2.resize(self.segmentation_mask, (w, h))
        overlay = image.copy()
        cell_overlay = np.zeros_like(image)
        cell_overlay[:, :, 1] = mask_resized
        overlay = cv2.addWeighted(overlay, 1.0, cell_overlay, 0.3, 0)

        # Draw contours
        contours, _ = cv2.findContours(
            mask_resized.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

        # Add anomaly heatmap if available and requested
        if show_heatmap and self._anomaly and self._anomaly.heatmap is not None:
            heatmap = self._anomaly.heatmap
            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_color = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(overlay, 0.7, heatmap_color, 0.3, 0)

        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)

        # Cell type and confidence
        cv2.putText(overlay, f"Cell: {self.cell_type} ({self.cell_type_confidence:.1%})",
                   (10, 30), font, 0.7, color, 2)

        # Confluency
        cv2.putText(overlay, f"Confluency: {self.confluency:.1f}% ({self.cell_count} cells)",
                   (10, 60), font, 0.6, color, 2)

        # Health status
        status_color = (0, 255, 0) if self.health_status == 'healthy' else (0, 165, 255)
        cv2.putText(overlay, f"Status: {self.health_status.upper()}",
                   (10, 90), font, 0.6, status_color, 2)

        return overlay

    def save_mask(self, path: str):
        """Save segmentation mask to file."""
        cv2.imwrite(path, self.segmentation_mask)

    def save_visualization(self, path: str, show_heatmap: bool = True):
        """Save visualization to file."""
        vis = self.visualize(show_heatmap)
        # Convert RGB to BGR for cv2
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# =============================================================================
# Model Architectures
# =============================================================================

class EfficientNetClassifier(nn.Module):
    """EfficientNet-B3 classifier matching training architecture."""

    CELL_TYPES = [
        'A172', 'BT474', 'BV2', 'HEK293', 'HeLa', 'Hepatocyte',
        'Huh7', 'MCF7', 'SHSY5Y', 'SKOV3', 'SkBr3', 'U2OS', 'U373'
    ]

    def __init__(self, num_classes: int = 13, pretrained: bool = False):
        super().__init__()

        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                'efficientnet_b3',
                pretrained=pretrained,
                num_classes=0,
                global_pool='avg'
            )
            self.feature_dim = self.backbone.num_features  # 1536
        else:
            # Fallback simple CNN
            self.backbone = self._create_fallback_backbone()
            self.feature_dim = 512

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def _create_fallback_backbone(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        features = self.backbone(x)
        if not TIMM_AVAILABLE:
            features = features.view(features.size(0), -1)
        return self.classifier(features)


class UNetSegmentation(nn.Module):
    """U-Net with ResNet-34 encoder matching training architecture."""

    def __init__(self):
        super().__init__()

        if SMP_AVAILABLE:
            self.model = smp.Unet(
                encoder_name='resnet34',
                encoder_weights=None,  # We'll load our own
                in_channels=3,
                classes=1,
                activation=None
            )
        else:
            # Fallback simple U-Net
            self.model = self._create_fallback_unet()

    def _create_fallback_unet(self):
        """Simple U-Net fallback."""
        class SimpleUNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = self._block(3, 64)
                self.enc2 = self._block(64, 128)
                self.enc3 = self._block(128, 256)
                self.enc4 = self._block(256, 512)

                # Decoder
                self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.dec3 = self._block(512, 256)
                self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2 = self._block(256, 128)
                self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1 = self._block(128, 64)

                self.final = nn.Conv2d(64, 1, 1)
                self.pool = nn.MaxPool2d(2)

            def _block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))

                d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

                return self.final(d1)

        return SimpleUNet()

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    """Convolutional block matching anomaly_detection.py architecture."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """Transposed convolution block matching anomaly_detection.py architecture."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class AnomalyAutoencoder(nn.Module):
    """
    Convolutional Autoencoder matching anomaly_detection.py architecture.
    Input: 128x128 RGB image
    Output: Reconstructed image + latent representation
    """

    def __init__(self, input_size: int = 128, latent_dim: int = 256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Encoder: 128x128 -> 4x4
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            ConvBlock(3, 32, kernel_size=4, stride=2, padding=1),
            # 64x64 -> 32x32
            ConvBlock(32, 64, kernel_size=4, stride=2, padding=1),
            # 32x32 -> 16x16
            ConvBlock(64, 128, kernel_size=4, stride=2, padding=1),
            # 16x16 -> 8x8
            ConvBlock(128, 256, kernel_size=4, stride=2, padding=1),
            # 8x8 -> 4x4
            ConvBlock(256, 512, kernel_size=4, stride=2, padding=1),
        )

        # Bottleneck (latent space)
        self.fc_encode = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)

        # Decoder: 4x4 -> 128x128
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            DeconvBlock(512, 256, kernel_size=4, stride=2, padding=1),
            # 8x8 -> 16x16
            DeconvBlock(256, 128, kernel_size=4, stride=2, padding=1),
            # 16x16 -> 32x32
            DeconvBlock(128, 64, kernel_size=4, stride=2, padding=1),
            # 32x32 -> 64x64
            DeconvBlock(64, 32, kernel_size=4, stride=2, padding=1),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent vector."""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        latent = self.fc_encode(features)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        features = self.fc_decode(z)
        features = features.view(features.size(0), 512, 4, 4)
        reconstructed = self.decoder(features)
        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns (reconstructed, latent)."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


# =============================================================================
# Main CellDiagnoseAI Class
# =============================================================================

class CellDiagnoseAI:
    """
    Unified inference pipeline for cell microscopy diagnostics.

    Combines three AI models:
    - Classification: EfficientNet-B3 for cell type identification (13 classes)
    - Segmentation: U-Net with ResNet-34 encoder for cell segmentation
    - Anomaly Detection: Convolutional Autoencoder for health assessment
    """

    # Cell type labels
    CELL_TYPES = [
        'A172', 'BT474', 'BV2', 'HEK293', 'HeLa', 'Hepatocyte',
        'Huh7', 'MCF7', 'SHSY5Y', 'SKOV3', 'SkBr3', 'U2OS', 'U373'
    ]

    # ImageNet normalization
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, device: Optional[str] = None):
        """
        Initialize CellDiagnoseAI.

        Args:
            device: 'cuda', 'cpu', or None for auto-detection
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize models (will be loaded later)
        self.classifier = None
        self.segmenter = None
        self.anomaly_detector = None

        # Model status
        self._classifier_loaded = False
        self._segmenter_loaded = False
        self._anomaly_loaded = False

        # Configuration
        self.class_to_idx = {name: idx for idx, name in enumerate(self.CELL_TYPES)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        # Anomaly baseline
        self.anomaly_baseline = 0.02

        print(f"[CellDiagnoseAI] Initialized on {self.device}")

    @classmethod
    def load(cls, checkpoint_dir: str, device: Optional[str] = None) -> 'CellDiagnoseAI':
        """
        Load pretrained models from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing model checkpoints
            device: Computing device

        Returns:
            Initialized CellDiagnoseAI instance
        """
        instance = cls(device=device)
        checkpoint_dir = Path(checkpoint_dir)

        # Load classification model
        classifier_path = checkpoint_dir / 'classifier_best.pth'
        if classifier_path.exists():
            instance._load_classifier(str(classifier_path))
        else:
            print(f"[Warning] Classifier checkpoint not found: {classifier_path}")

        # Load segmentation model
        segmentation_path = checkpoint_dir / 'segmentation_best.pth'
        if segmentation_path.exists():
            instance._load_segmenter(str(segmentation_path))
        else:
            print(f"[Warning] Segmentation checkpoint not found: {segmentation_path}")

        # Load anomaly detection model
        anomaly_path = checkpoint_dir / 'anomaly_detector.pth'
        if anomaly_path.exists():
            instance._load_anomaly_detector(str(anomaly_path))
        else:
            print(f"[Warning] Anomaly detector checkpoint not found: {anomaly_path}")

        return instance

    def _load_classifier(self, path: str):
        """Load classification model."""
        self.classifier = EfficientNetClassifier(num_classes=len(self.CELL_TYPES))

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Update class mapping if available
            if 'class_to_idx' in checkpoint:
                self.class_to_idx = checkpoint['class_to_idx']
                self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        else:
            state_dict = checkpoint

        self.classifier.load_state_dict(state_dict)
        self.classifier.to(self.device)
        self.classifier.eval()
        self._classifier_loaded = True
        print(f"[+] Classifier loaded from {path}")

    def _load_segmenter(self, path: str):
        """Load segmentation model."""
        self.segmenter = UNetSegmentation()

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle Lightning checkpoint format
        if 'state_dict' in checkpoint:
            # Lightning format: remove 'model.' prefix
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    state_dict[k[6:]] = v
                else:
                    state_dict[k] = v
            self.segmenter.model.load_state_dict(state_dict)
        elif 'model_state_dict' in checkpoint:
            self.segmenter.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading directly
            try:
                self.segmenter.model.load_state_dict(checkpoint)
            except:
                self.segmenter.load_state_dict(checkpoint)

        self.segmenter.to(self.device)
        self.segmenter.eval()
        self._segmenter_loaded = True
        print(f"[+] Segmenter loaded from {path}")

    def _load_anomaly_detector(self, path: str):
        """Load anomaly detection model."""
        self.anomaly_detector = AnomalyAutoencoder(latent_dim=256)

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            self.anomaly_detector.load_state_dict(checkpoint['model_state_dict'])
            self.anomaly_baseline = checkpoint.get('baseline_error', 0.02)
        else:
            self.anomaly_detector.load_state_dict(checkpoint)

        self.anomaly_detector.to(self.device)
        self.anomaly_detector.eval()
        self._anomaly_loaded = True
        print(f"[+] Anomaly detector loaded from {path}")

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    def _load_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Load and convert image to RGB numpy array."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return image

    def _preprocess_classification(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for classification (300x300)."""
        # Resize
        resized = cv2.resize(image, (300, 300))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.MEAN) / self.STD

        # To tensor (B, C, H, W)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device)

    def _preprocess_segmentation(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for segmentation (512x512)."""
        # Resize
        resized = cv2.resize(image, (512, 512))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.MEAN) / self.STD

        # To tensor
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device)

    def _preprocess_anomaly(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for anomaly detection (128x128)."""
        # Resize
        resized = cv2.resize(image, (128, 128))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # To tensor
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device)

    # -------------------------------------------------------------------------
    # Individual Predictions
    # -------------------------------------------------------------------------

    def classify(self, image: Union[str, np.ndarray]) -> ClassificationResult:
        """
        Classify cell type.

        Args:
            image: Input image (path, numpy array, or PIL Image)

        Returns:
            ClassificationResult with cell type and confidence
        """
        image = self._load_image(image)

        if not self._classifier_loaded:
            # Return mock result
            return ClassificationResult(
                cell_type="Unknown",
                confidence=0.0,
                all_probabilities={ct: 0.0 for ct in self.CELL_TYPES},
                health_status="unknown",
                health_confidence=0.0
            )

        tensor = self._preprocess_classification(image)

        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Get prediction
        pred_idx = np.argmax(probs)
        pred_class = self.idx_to_class.get(pred_idx, "Unknown")
        confidence = float(probs[pred_idx])

        # Build probability dict
        all_probs = {}
        for idx, prob in enumerate(probs):
            class_name = self.idx_to_class.get(idx, f"Class_{idx}")
            all_probs[class_name] = float(prob)

        return ClassificationResult(
            cell_type=pred_class,
            confidence=confidence,
            all_probabilities=all_probs,
            health_status="healthy" if confidence > 0.8 else "uncertain",
            health_confidence=confidence
        )

    def segment(self, image: Union[str, np.ndarray]) -> SegmentationResult:
        """
        Segment cells in image.

        Args:
            image: Input image

        Returns:
            SegmentationResult with mask, confluency, and cell count
        """
        image = self._load_image(image)
        original_h, original_w = image.shape[:2]

        if not self._segmenter_loaded:
            # Fallback to OpenCV-based segmentation
            return self._fallback_segmentation(image)

        tensor = self._preprocess_segmentation(image)

        with torch.no_grad():
            logits = self.segmenter(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Threshold and resize to original
        mask = (probs > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (original_w, original_h))

        # Calculate confluency
        confluency = float((mask > 0).sum() / mask.size * 100)

        # Count cells using connected components
        num_labels, labeled = cv2.connectedComponents(mask)
        num_cells = num_labels - 1  # Subtract background

        # Create overlay
        overlay = image.copy()
        cell_overlay = np.zeros_like(image)
        cell_overlay[:, :, 1] = mask
        overlay = cv2.addWeighted(overlay, 1.0, cell_overlay, 0.3, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

        return SegmentationResult(
            mask=mask,
            confluency_percent=round(confluency, 2),
            cell_count=num_cells,
            overlay=overlay
        )

    def _fallback_segmentation(self, image: np.ndarray) -> SegmentationResult:
        """OpenCV-based fallback segmentation."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Adaptive threshold
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Calculate metrics
        confluency = float((mask > 0).sum() / mask.size * 100)
        num_labels, labeled = cv2.connectedComponents(mask)
        num_cells = num_labels - 1

        # Create overlay
        overlay = image.copy()
        cell_overlay = np.zeros_like(image)
        cell_overlay[:, :, 1] = mask
        overlay = cv2.addWeighted(overlay, 1.0, cell_overlay, 0.3, 0)

        return SegmentationResult(
            mask=mask,
            confluency_percent=round(confluency, 2),
            cell_count=num_cells,
            overlay=overlay
        )

    def detect_anomalies(self, image: Union[str, np.ndarray]) -> AnomalyResult:
        """
        Detect anomalies/contamination.

        Args:
            image: Input image

        Returns:
            AnomalyResult with score, severity, and possible issues
        """
        image = self._load_image(image)
        original_h, original_w = image.shape[:2]

        if not self._anomaly_loaded:
            # Return neutral result
            return AnomalyResult(
                score=0.0,
                normalized_score=1.0,
                is_anomaly=False,
                severity="normal",
                heatmap=None,
                possible_issues=[],
                recommendation="Model not loaded. Cannot assess anomalies."
            )

        tensor = self._preprocess_anomaly(image)

        with torch.no_grad():
            reconstructed, latent = self.anomaly_detector(tensor)
            error_map = (tensor - reconstructed).pow(2).mean(dim=1)
            score = error_map.mean().item()

        # Normalize score
        normalized = score / max(self.anomaly_baseline, 0.01)

        # Determine severity
        if normalized < 1.5:
            severity = "normal"
        elif normalized < 3.0:
            severity = "low"
        elif normalized < 5.0:
            severity = "medium"
        elif normalized < 10.0:
            severity = "high"
        else:
            severity = "critical"

        is_anomaly = severity != "normal"

        # Create heatmap
        heatmap = error_map.squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (original_w, original_h))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Identify issues
        issues = self._identify_issues(heatmap, normalized)
        recommendation = self._get_recommendation(severity, issues)

        return AnomalyResult(
            score=score,
            normalized_score=normalized,
            is_anomaly=is_anomaly,
            severity=severity,
            heatmap=heatmap,
            possible_issues=issues,
            recommendation=recommendation
        )

    def _identify_issues(self, heatmap: np.ndarray, score: float) -> List[str]:
        """Identify possible issues from anomaly analysis."""
        issues = []

        if score < 1.5:
            return issues

        mean_err = heatmap.mean()
        std_err = heatmap.std()
        max_err = heatmap.max()

        if std_err > mean_err * 2:
            if max_err > mean_err * 5:
                issues.append("bacterial_contamination")
            else:
                issues.append("debris")

        if std_err < mean_err * 0.5 and score > 3.0:
            issues.append("cell_stress")

        if score > 5.0:
            issues.append("dead_cells")

        if 1.5 < score < 3.0 and std_err < mean_err:
            issues.append("mycoplasma_possible")

        return issues

    def _get_recommendation(self, severity: str, issues: List[str]) -> str:
        """Generate recommendation based on findings."""
        if severity == "normal":
            return "Culture appears healthy. Continue normal monitoring."
        elif severity == "low":
            return "Minor abnormalities detected. Monitor closely and recheck in 24h."
        elif severity == "medium":
            if "bacterial" in str(issues):
                return "Possible contamination. Examine under higher magnification."
            return "Abnormalities detected. Check culture conditions."
        elif severity == "high":
            return "Significant issues detected. Consider discarding culture."
        else:
            return "CRITICAL: Major contamination/cell death. Discard and sterilize."

    # -------------------------------------------------------------------------
    # Complete Diagnosis
    # -------------------------------------------------------------------------

    def diagnose(self, image: Union[str, np.ndarray, Path, Image.Image]) -> DiagnosisResult:
        """
        Run complete diagnostic pipeline.

        Args:
            image: Input image (file path, numpy array, or PIL Image)

        Returns:
            DiagnosisResult with all analysis results
        """
        # Load image
        img_array = self._load_image(image)
        h, w, c = img_array.shape

        # Run all analyses
        classification = self.classify(img_array)
        segmentation = self.segment(img_array)
        anomaly = self.detect_anomalies(img_array)

        # Determine overall health status
        health_factors = []

        if segmentation.confluency_percent < 20:
            health_factors.append("low_confluency")
        elif segmentation.confluency_percent > 95:
            health_factors.append("over_confluent")

        if anomaly.is_anomaly:
            health_factors.append("anomaly_detected")

        if classification.confidence < 0.5:
            health_factors.append("uncertain_classification")

        if not health_factors:
            health_status = "healthy"
        elif len(health_factors) == 1 and health_factors[0] in ["low_confluency", "over_confluent"]:
            health_status = "needs_attention"
        else:
            health_status = "unhealthy"

        # Compile recommendation
        recommendations = []
        if "low_confluency" in health_factors:
            recommendations.append("Allow cells to grow before passaging.")
        if "over_confluent" in health_factors:
            recommendations.append("Passage cells soon to prevent stress.")
        if anomaly.recommendation:
            recommendations.append(anomaly.recommendation)

        recommendation = " ".join(recommendations) if recommendations else "Continue normal monitoring."

        return DiagnosisResult(
            image_shape=(h, w, c),
            timestamp=datetime.now().isoformat(),
            cell_type=classification.cell_type,
            cell_type_confidence=classification.confidence,
            cell_type_probabilities=classification.all_probabilities,
            confluency=segmentation.confluency_percent,
            cell_count=segmentation.cell_count,
            segmentation_mask=segmentation.mask,
            health_status=health_status,
            anomaly_score=anomaly.score,
            anomaly_severity=anomaly.severity,
            possible_issues=anomaly.possible_issues,
            recommendation=recommendation,
            _classification=classification,
            _segmentation=segmentation,
            _anomaly=anomaly,
            _original_image=img_array
        )

    def __repr__(self):
        status = []
        status.append(f"classifier={'loaded' if self._classifier_loaded else 'not loaded'}")
        status.append(f"segmenter={'loaded' if self._segmenter_loaded else 'not loaded'}")
        status.append(f"anomaly_detector={'loaded' if self._anomaly_loaded else 'not loaded'}")
        return f"CellDiagnoseAI({', '.join(status)}, device={self.device})"


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description='CellDiagnose-AI: Cell Microscopy Diagnostics')
    parser.add_argument('image', type=str, help='Path to microscopy image')
    parser.add_argument('--checkpoint-dir', '-c', type=str, default='checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for JSON report')
    parser.add_argument('--visualize', '-v', type=str, default=None,
                       help='Output path for visualization image')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Load model
    print(f"Loading models from {args.checkpoint_dir}...")
    model = CellDiagnoseAI.load(args.checkpoint_dir, device=args.device)
    print(model)

    # Run diagnosis
    print(f"\nAnalyzing {args.image}...")
    result = model.diagnose(args.image)

    # Print results
    print("\n" + "=" * 60)
    print("DIAGNOSIS RESULT")
    print("=" * 60)
    print(f"Cell Type:     {result.cell_type} ({result.cell_type_confidence:.1%})")
    print(f"Confluency:    {result.confluency:.1f}% ({result.cell_count} cells)")
    print(f"Health Status: {result.health_status.upper()}")
    print(f"Anomaly Score: {result.anomaly_score:.4f} (severity: {result.anomaly_severity})")
    if result.possible_issues:
        print(f"Issues:        {', '.join(result.possible_issues)}")
    print(f"\nRecommendation: {result.recommendation}")
    print("=" * 60)

    # Save outputs
    if args.output:
        result.to_json(args.output)
        print(f"\nReport saved to: {args.output}")

    if args.visualize:
        result.save_visualization(args.visualize)
        print(f"Visualization saved to: {args.visualize}")


if __name__ == '__main__':
    main()
