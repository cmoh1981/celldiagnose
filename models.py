"""
CellDiagnose-AI: Deep Learning Model Architecture
=================================================
PyTorch model classes with pluggable architecture for easy weight swapping.
Currently returns mock predictions; ready for real trained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import cv2


# =============================================================================
# Abstract Base Classes
# =============================================================================

class BaseModel(ABC):
    """Abstract base class for all diagnostic models."""

    def __init__(self, device: str = None):
        """
        Initialize model with specified device.

        Args:
            device: 'cuda', 'cpu', or None for auto-detection
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_weights(self, weights_path: str) -> bool:
        """Load trained weights from file."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run inference on input image."""
        pass

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Default preprocessing pipeline."""
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))

        # Normalize to ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        normalized = (resized / 255.0 - mean) / std

        # Convert to tensor (B, C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()

        return tensor.to(self.device)


# =============================================================================
# Segmentation Models (U-Net Architecture)
# =============================================================================

class UNetBlock(nn.Module):
    """Double convolution block for U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for cell segmentation.
    Paper: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 features: list = [64, 128, 256, 512]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.upconvs = nn.ModuleList()

        # Encoder path
        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(UNetBlock(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)

        # Decoder path
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(UNetBlock(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            skip = skip_connections[idx]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return torch.sigmoid(self.final_conv(x))


class SegmentationModel(BaseModel):
    """
    Wrapper for U-Net segmentation model.
    Currently returns mock predictions; swap weights for real inference.
    """

    # Class variable to track weight loading status message
    MOCK_MODE_MSG = "Running in MOCK mode - load trained weights for real predictions"

    def __init__(self, device: str = None):
        super().__init__(device)
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        self.input_size = (256, 256)

    def load_weights(self, weights_path: str) -> bool:
        """
        Load trained U-Net weights.

        Args:
            weights_path: Path to .pth file with trained weights

        Returns:
            True if weights loaded successfully
        """
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
            return False

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for segmentation."""
        resized = cv2.resize(image, self.input_size)
        normalized = resized / 255.0

        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Generate segmentation mask for input image.

        Args:
            image: RGB numpy array

        Returns:
            Dictionary with 'mask', 'confluency_percent', 'is_mock'
        """
        original_size = image.shape[:2]

        if self.is_loaded:
            # Real inference with trained model
            self.model.eval()
            with torch.no_grad():
                input_tensor = self.preprocess(image)
                output = self.model(input_tensor)

                # Convert to numpy mask
                mask = output.squeeze().cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8) * 255

                # Resize back to original
                mask = cv2.resize(mask, (original_size[1], original_size[0]))

            confluency = (np.count_nonzero(mask) / mask.size) * 100
            is_mock = False
        else:
            # Mock prediction using simple thresholding
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            _, mask = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            confluency = (np.count_nonzero(mask) / mask.size) * 100
            is_mock = True

        return {
            'mask': mask,
            'confluency_percent': round(confluency, 2),
            'is_mock': is_mock,
            'model_status': self.MOCK_MODE_MSG if is_mock else "Using trained weights"
        }


# =============================================================================
# Classification Models (EfficientNet / ResNet)
# =============================================================================

class EfficientNetB0(nn.Module):
    """
    Simplified EfficientNet-B0 inspired architecture.
    For production, use torchvision.models.efficientnet_b0
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()

        # Simplified feature extractor
        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            # Blocks (simplified)
            self._make_block(32, 16, 1),
            self._make_block(16, 24, 2),
            self._make_block(24, 40, 2),
            self._make_block(40, 80, 2),
            self._make_block(80, 112, 1),
            self._make_block(112, 192, 2),
            self._make_block(192, 320, 1),

            # Head
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def _make_block(self, in_ch: int, out_ch: int, stride: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet50Classifier(nn.Module):
    """
    Simplified ResNet-50 inspired architecture.
    For production, use torchvision.models.resnet50
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()

        # Simplified implementation
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual layers (simplified)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int = 1):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks - 1):
            layers.extend([
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CellClassifier(BaseModel):
    """
    Cell type and health classification model.
    Supports both EfficientNet-B0 and ResNet50 backbones.
    """

    # Define class labels - dynamically loaded from config
    # Default cell types (updated from all available datasets)
    CELL_TYPES = [
        'A172',    # Glioblastoma
        'BT474',   # Breast cancer
        'BV2',     # Microglia (mouse)
        'HeLa',    # Cervical cancer
        'Huh7',    # Hepatocellular carcinoma
        'MCF7',    # Breast cancer
        'SHSY5Y',  # Neuroblastoma
        'SKOV3',   # Ovarian cancer
        'SkBr3',   # Breast cancer
        'U373',    # Glioblastoma astrocytoma
        'Unknown'  # Fallback
    ]
    HEALTH_STATUS = ['Healthy', 'Unhealthy']

    @classmethod
    def load_cell_types_from_config(cls):
        """Load cell types from dataset config if available."""
        try:
            from config import get_config
            config = get_config()
            if config.data.cell_types:
                cls.CELL_TYPES = config.data.cell_types + ['Unknown']
        except Exception:
            pass  # Use defaults

    def __init__(self, backbone: str = 'efficientnet', device: str = None):
        """
        Initialize classifier with specified backbone.

        Args:
            backbone: 'efficientnet' or 'resnet50'
            device: Computing device
        """
        super().__init__(device)
        self.backbone_name = backbone

        # Number of outputs: cell types + health status
        num_cell_types = len(self.CELL_TYPES) - 1  # Exclude 'Unknown'
        num_health = len(self.HEALTH_STATUS)

        if backbone == 'efficientnet':
            self.model = EfficientNetB0(num_classes=num_cell_types + num_health)
        else:
            self.model = ResNet50Classifier(num_classes=num_cell_types + num_health)

        self.model = self.model.to(self.device)
        self.num_cell_types = num_cell_types

    def load_weights(self, weights_path: str) -> bool:
        """Load trained classification weights."""
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
            return False

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify cell type and health status.

        Args:
            image: RGB numpy array

        Returns:
            Dictionary with cell_type, health_status, and confidences
        """
        if self.is_loaded:
            # Real inference
            self.model.eval()
            with torch.no_grad():
                input_tensor = self.preprocess(image)
                output = self.model(input_tensor)

                # Split output into cell type and health predictions
                cell_logits = output[0, :self.num_cell_types]
                health_logits = output[0, self.num_cell_types:]

                cell_probs = F.softmax(cell_logits, dim=0).cpu().numpy()
                health_probs = F.softmax(health_logits, dim=0).cpu().numpy()

                cell_idx = np.argmax(cell_probs)
                health_idx = np.argmax(health_probs)

            return {
                'cell_type': self.CELL_TYPES[cell_idx],
                'cell_type_confidence': float(cell_probs[cell_idx]),
                'cell_type_probs': {self.CELL_TYPES[i]: float(p)
                                    for i, p in enumerate(cell_probs)},
                'health_status': self.HEALTH_STATUS[health_idx],
                'health_confidence': float(health_probs[health_idx]),
                'is_mock': False,
                'model_status': f"Using trained {self.backbone_name} weights"
            }
        else:
            # Mock prediction with realistic distribution
            return self._generate_mock_prediction(image)

    def _generate_mock_prediction(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Generate realistic mock predictions based on image features.
        Uses basic image statistics to vary predictions.
        """
        # Use image statistics to generate semi-deterministic mock results
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # Generate pseudo-random but deterministic confidences based on image
        np.random.seed(int(mean_intensity * 100) % 10000)

        # Cell type prediction (mock)
        cell_probs = np.random.dirichlet(np.ones(self.num_cell_types) * 2)
        cell_idx = np.argmax(cell_probs)

        # Health prediction - base on image variance (mock heuristic)
        # Higher variance often indicates healthier, actively growing cells
        if std_intensity > 40:
            health_probs = np.array([0.75 + np.random.uniform(0, 0.15),
                                     0.10 + np.random.uniform(0, 0.05)])
        else:
            health_probs = np.array([0.30 + np.random.uniform(0, 0.10),
                                     0.60 + np.random.uniform(0, 0.10)])
        health_probs = health_probs / health_probs.sum()
        health_idx = np.argmax(health_probs)

        return {
            'cell_type': self.CELL_TYPES[cell_idx],
            'cell_type_confidence': float(cell_probs[cell_idx]),
            'cell_type_probs': {self.CELL_TYPES[i]: float(p)
                                for i, p in enumerate(cell_probs)},
            'health_status': self.HEALTH_STATUS[health_idx],
            'health_confidence': float(health_probs[health_idx]),
            'is_mock': True,
            'model_status': f"MOCK MODE - Load trained {self.backbone_name} weights for real predictions"
        }


# =============================================================================
# Anomaly Detection Autoencoder
# =============================================================================

class AnomalyAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection.
    Trained on healthy cell images; anomalies produce high reconstruction error.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 128 -> 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),  # 8 -> 4
            nn.LeakyReLU(0.2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly scoring."""
        reconstructed = self.forward(x)
        error = F.mse_loss(reconstructed, x, reduction='none')
        return error.mean(dim=(1, 2, 3))  # Per-sample error


class AnomalyDetectionModel(BaseModel):
    """
    Autoencoder-based anomaly detection for contamination screening.
    """

    def __init__(self, threshold: float = 0.05, device: str = None):
        """
        Initialize anomaly detector.

        Args:
            threshold: Reconstruction error threshold for anomaly flag
            device: Computing device
        """
        super().__init__(device)
        self.model = AnomalyAutoencoder(latent_dim=128).to(self.device)
        self.threshold = threshold
        self.input_size = (128, 128)

    def load_weights(self, weights_path: str) -> bool:
        """Load trained autoencoder weights."""
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
            return False

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess for autoencoder input."""
        resized = cv2.resize(image, self.input_size)
        normalized = resized / 255.0

        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies in cell image.

        Args:
            image: RGB numpy array

        Returns:
            Dictionary with anomaly_score, is_anomalous, reconstruction
        """
        if self.is_loaded:
            self.model.eval()
            with torch.no_grad():
                input_tensor = self.preprocess(image)
                reconstruction_error = self.model.get_reconstruction_error(input_tensor)
                error_score = float(reconstruction_error.item())

                # Get reconstruction for visualization
                reconstructed = self.model(input_tensor)
                recon_image = reconstructed.squeeze().permute(1, 2, 0).cpu().numpy()
                recon_image = (recon_image * 255).astype(np.uint8)

            return {
                'anomaly_score': error_score,
                'is_anomalous': error_score > self.threshold,
                'reconstruction': recon_image,
                'is_mock': False,
                'model_status': "Using trained autoencoder weights"
            }
        else:
            # Fall back to texture-based analysis (see utils.py)
            return {
                'anomaly_score': None,
                'is_anomalous': None,
                'reconstruction': None,
                'is_mock': True,
                'model_status': "MOCK MODE - Using texture analysis fallback"
            }


# =============================================================================
# Model Loader / Factory
# =============================================================================

class ModelLoader:
    """
    Factory class for loading and managing diagnostic models.
    Centralizes model initialization and weight loading.
    """

    # Registry of available models
    AVAILABLE_MODELS = {
        'segmentation': {
            'unet': SegmentationModel,
        },
        'classification': {
            'efficientnet': lambda device: CellClassifier('efficientnet', device),
            'resnet50': lambda device: CellClassifier('resnet50', device),
        },
        'anomaly': {
            'autoencoder': AnomalyDetectionModel,
        }
    }

    def __init__(self, device: str = None):
        """
        Initialize model loader.

        Args:
            device: 'cuda', 'cpu', or None for auto-detection
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._loaded_models = {}

    def load_model(self, model_type: str, model_name: str,
                   weights_path: Optional[str] = None) -> BaseModel:
        """
        Load a model by type and name.

        Args:
            model_type: 'segmentation', 'classification', or 'anomaly'
            model_name: Specific model architecture name
            weights_path: Optional path to trained weights

        Returns:
            Initialized model instance
        """
        cache_key = f"{model_type}_{model_name}"

        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_name not in self.AVAILABLE_MODELS[model_type]:
            raise ValueError(f"Unknown model: {model_name} for type {model_type}")

        # Initialize model
        model_factory = self.AVAILABLE_MODELS[model_type][model_name]
        model = model_factory(self.device)

        # Load weights if provided
        if weights_path:
            model.load_weights(weights_path)

        self._loaded_models[cache_key] = model
        return model

    def get_segmentation_model(self, model_name: str = 'unet',
                               weights_path: str = None) -> SegmentationModel:
        """Convenience method for segmentation models."""
        return self.load_model('segmentation', model_name, weights_path)

    def get_classification_model(self, model_name: str = 'efficientnet',
                                  weights_path: str = None) -> CellClassifier:
        """Convenience method for classification models."""
        return self.load_model('classification', model_name, weights_path)

    def get_anomaly_model(self, model_name: str = 'autoencoder',
                          weights_path: str = None) -> AnomalyDetectionModel:
        """Convenience method for anomaly detection models."""
        return self.load_model('anomaly', model_name, weights_path)

    def list_available_models(self) -> Dict[str, list]:
        """Return dictionary of all available model types and names."""
        return {
            model_type: list(models.keys())
            for model_type, models in self.AVAILABLE_MODELS.items()
        }

    @property
    def device_info(self) -> str:
        """Return current device information."""
        if self.device == 'cuda':
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        return "CPU"
