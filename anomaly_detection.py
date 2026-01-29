#!/usr/bin/env python3
"""
CellDiagnose-AI: Anomaly Detection Module
==========================================
Convolutional Autoencoder-based anomaly detection for cell health monitoring.

Strategy:
- Train ONLY on healthy/normal cell images
- Detect anomalies by measuring reconstruction error
- High error = Contamination, Dead cells, Stress, Debris
- Low error = Normal/Healthy

Detectable Conditions:
- Bacterial contamination (black dots/rods in background)
- Dead/floating cells (round, refractile)
- Starvation/stress (vacuoles, abnormal shapes)
- Debris and artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2
from PIL import Image


# =============================================================================
# Convolutional Autoencoder Architecture
# =============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """Transposed convolution block for decoder."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class CellAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for Cell Anomaly Detection.

    Trained on healthy cells only. Anomalies produce high reconstruction error.

    Architecture:
        Encoder: 3 -> 32 -> 64 -> 128 -> 256 (latent)
        Decoder: 256 -> 128 -> 64 -> 32 -> 3

    Input: 128x128 RGB image patches
    Output: Reconstructed image + Anomaly score
    """

    def __init__(self, input_size: int = 128, latent_dim: int = 256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Encoder
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

        # Decoder
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
        """
        Forward pass.

        Returns:
            reconstructed: Reconstructed image
            latent: Latent representation
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate pixel-wise reconstruction error.

        Returns:
            error_map: Per-pixel MSE error (for heatmap visualization)
        """
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error_map = (x - reconstructed).pow(2).mean(dim=1)  # Average over channels
        return error_map

    def get_anomaly_score(self, x: torch.Tensor) -> float:
        """
        Calculate overall anomaly score for an image.

        Returns:
            score: Single scalar (higher = more anomalous)
        """
        error_map = self.get_reconstruction_error(x)
        return error_map.mean().item()


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for more robust anomaly detection.
    Uses KL divergence to regularize latent space.
    """

    def __init__(self, input_size: int = 128, latent_dim: int = 256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, kernel_size=4, stride=2, padding=1),
            ConvBlock(32, 64, kernel_size=4, stride=2, padding=1),
            ConvBlock(64, 128, kernel_size=4, stride=2, padding=1),
            ConvBlock(128, 256, kernel_size=4, stride=2, padding=1),
            ConvBlock(256, 512, kernel_size=4, stride=2, padding=1),
        )

        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)

        # Decoder
        self.decoder = nn.Sequential(
            DeconvBlock(512, 256, kernel_size=4, stride=2, padding=1),
            DeconvBlock(256, 128, kernel_size=4, stride=2, padding=1),
            DeconvBlock(128, 64, kernel_size=4, stride=2, padding=1),
            DeconvBlock(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.fc_decode(z)
        features = features.view(features.size(0), 512, 4, 4)
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def get_anomaly_score(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            reconstructed, mu, logvar = self.forward(x)
            recon_error = F.mse_loss(reconstructed, x, reduction='mean')
            # Add KL divergence component
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return (recon_error + 0.1 * kl_div).item()


# =============================================================================
# Anomaly Detector Class (High-level API)
# =============================================================================

class CellAnomalyDetector:
    """
    High-level API for cell anomaly detection.

    Usage:
        detector = CellAnomalyDetector()
        detector.load_model('checkpoints/anomaly_detector.pth')

        result = detector.analyze(image)
        # result = {
        #     'anomaly_score': 0.15,
        #     'is_anomaly': True,
        #     'anomaly_map': np.array(...),
        #     'confidence': 0.87,
        #     'possible_issues': ['Contamination', 'Debris']
        # }
    """

    # Thresholds for different conditions
    THRESHOLDS = {
        'low': 0.05,      # Slightly abnormal
        'medium': 0.10,   # Clearly abnormal
        'high': 0.20,     # Severe anomaly
        'critical': 0.35  # Major contamination
    }

    # Issue descriptions
    ISSUE_DESCRIPTIONS = {
        'bacterial': 'Possible bacterial contamination detected (small dark particles)',
        'dead_cells': 'Dead or floating cells detected (bright, round objects)',
        'stress': 'Cell stress indicators (vacuoles, abnormal morphology)',
        'debris': 'Debris or artifacts in the image',
        'mycoplasma': 'Subtle changes detected - consider PCR test for mycoplasma',
        'overgrowth': 'Possible overgrowth or confluency stress'
    }

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize anomaly detector.

        Args:
            model_path: Path to trained model weights
            device: 'cuda', 'cpu', or 'auto'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = CellAutoencoder(input_size=128, latent_dim=256)
        self.model.to(self.device)
        self.model.eval()

        self.baseline_error = 0.02  # Expected error for healthy cells
        self.is_trained = False

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load trained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            self.baseline_error = checkpoint.get('baseline_error', 0.02)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.is_trained = True
        print(f"[+] Anomaly detection model loaded from {model_path}")

    def save_model(self, model_path: str):
        """Save model weights."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'baseline_error': self.baseline_error
        }, model_path)
        print(f"[+] Model saved to {model_path}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to 128x128
        if image.shape[:2] != (128, 128):
            image = cv2.resize(image, (128, 128))

        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    def analyze(self, image: np.ndarray) -> Dict:
        """
        Analyze image for anomalies.

        Args:
            image: Input image (numpy array, any size)

        Returns:
            Dictionary with:
                - anomaly_score: Overall score (0-1, higher = more anomalous)
                - is_anomaly: Boolean flag
                - severity: 'normal', 'low', 'medium', 'high', 'critical'
                - anomaly_map: Heatmap showing anomaly locations
                - reconstructed: Reconstructed image
                - possible_issues: List of potential problems
                - recommendation: Action recommendation
        """
        # Preprocess
        tensor = self.preprocess(image)

        with torch.no_grad():
            # Get reconstruction
            reconstructed, latent = self.model(tensor)

            # Calculate error map
            error_map = (tensor - reconstructed).pow(2).mean(dim=1)
            anomaly_score = error_map.mean().item()

            # Normalize score relative to baseline
            normalized_score = anomaly_score / max(self.baseline_error, 0.01)

        # Determine severity
        if normalized_score < 1.5:
            severity = 'normal'
        elif normalized_score < 3.0:
            severity = 'low'
        elif normalized_score < 5.0:
            severity = 'medium'
        elif normalized_score < 10.0:
            severity = 'high'
        else:
            severity = 'critical'

        is_anomaly = severity != 'normal'

        # Generate anomaly map (resize to original size)
        anomaly_map = error_map.squeeze().cpu().numpy()
        anomaly_map = cv2.resize(anomaly_map, (image.shape[1], image.shape[0]))

        # Normalize for visualization
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

        # Get reconstructed image
        recon_np = reconstructed.squeeze().cpu().numpy().transpose(1, 2, 0)
        recon_np = (recon_np * 255).astype(np.uint8)
        recon_np = cv2.resize(recon_np, (image.shape[1], image.shape[0]))

        # Identify possible issues based on error patterns
        possible_issues = self._identify_issues(error_map.squeeze().cpu().numpy(), normalized_score)

        # Generate recommendation
        recommendation = self._get_recommendation(severity, possible_issues)

        return {
            'anomaly_score': anomaly_score,
            'normalized_score': normalized_score,
            'is_anomaly': is_anomaly,
            'severity': severity,
            'anomaly_map': anomaly_map,
            'reconstructed': recon_np,
            'possible_issues': possible_issues,
            'recommendation': recommendation,
            'confidence': min(1.0, normalized_score / 10.0) if is_anomaly else 1.0 - normalized_score / 1.5
        }

    def _identify_issues(self, error_map: np.ndarray, score: float) -> List[str]:
        """Identify possible issues based on error patterns."""
        issues = []

        if score < 1.5:
            return issues

        # Analyze error distribution
        mean_error = error_map.mean()
        std_error = error_map.std()
        max_error = error_map.max()

        # High variance suggests localized issues (contamination, debris)
        if std_error > mean_error * 2:
            # Check for small bright spots (bacterial contamination)
            if max_error > mean_error * 5:
                issues.append('bacterial')
            else:
                issues.append('debris')

        # Uniform high error suggests general stress
        if std_error < mean_error * 0.5 and score > 3.0:
            issues.append('stress')

        # Very high score with moderate variance
        if score > 5.0:
            issues.append('dead_cells')

        # Subtle but consistent changes
        if 1.5 < score < 3.0 and std_error < mean_error:
            issues.append('mycoplasma')

        # High confluency patterns
        if score > 4.0 and std_error < mean_error * 0.8:
            issues.append('overgrowth')

        return issues

    def _get_recommendation(self, severity: str, issues: List[str]) -> str:
        """Generate action recommendation."""
        if severity == 'normal':
            return "Cell culture appears healthy. Continue normal monitoring."

        elif severity == 'low':
            return "Minor abnormalities detected. Monitor closely and check again in 24 hours."

        elif severity == 'medium':
            if 'bacterial' in issues:
                return "Possible contamination. Examine under higher magnification. Consider antibiotic treatment or discard."
            elif 'mycoplasma' in issues:
                return "Subtle changes detected. Recommend PCR test for mycoplasma contamination."
            else:
                return "Abnormalities detected. Check culture conditions (pH, temperature, media)."

        elif severity == 'high':
            if 'dead_cells' in issues:
                return "Significant cell death observed. Change media immediately and assess viability."
            else:
                return "Severe abnormalities. Consider discarding culture to prevent spread."

        else:  # critical
            return "CRITICAL: Major contamination or cell death. Discard culture and sterilize incubator."

    def analyze_full_image(self, image: np.ndarray, patch_size: int = 128,
                          overlap: int = 32) -> Dict:
        """
        Analyze full-size image by scanning with patches.

        Returns aggregated results and full-resolution anomaly map.
        """
        h, w = image.shape[:2]
        step = patch_size - overlap

        # Initialize full anomaly map
        full_anomaly_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        all_scores = []

        # Scan with patches
        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                patch = image[y:y+patch_size, x:x+patch_size]
                result = self.analyze(patch)

                all_scores.append(result['anomaly_score'])

                # Add to full map
                patch_map = cv2.resize(result['anomaly_map'], (patch_size, patch_size))
                full_anomaly_map[y:y+patch_size, x:x+patch_size] += patch_map
                count_map[y:y+patch_size, x:x+patch_size] += 1

        # Average overlapping regions
        count_map[count_map == 0] = 1
        full_anomaly_map /= count_map

        # Aggregate scores
        mean_score = np.mean(all_scores)
        max_score = np.max(all_scores)

        # Use max score for severity (be conservative)
        normalized_score = max_score / max(self.baseline_error, 0.01)

        if normalized_score < 1.5:
            severity = 'normal'
        elif normalized_score < 3.0:
            severity = 'low'
        elif normalized_score < 5.0:
            severity = 'medium'
        elif normalized_score < 10.0:
            severity = 'high'
        else:
            severity = 'critical'

        return {
            'anomaly_score': mean_score,
            'max_score': max_score,
            'normalized_score': normalized_score,
            'is_anomaly': severity != 'normal',
            'severity': severity,
            'anomaly_map': full_anomaly_map,
            'num_patches_analyzed': len(all_scores),
            'possible_issues': self._identify_issues(full_anomaly_map, normalized_score),
            'recommendation': self._get_recommendation(severity, [])
        }

    def create_visualization(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        Create visualization with anomaly heatmap overlay.

        Returns:
            Combined visualization image
        """
        h, w = image.shape[:2]

        # Create heatmap
        anomaly_map = result['anomaly_map']
        heatmap = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (w, h))

        # Ensure image is RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()

        # Blend
        alpha = 0.4
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)

        # Add severity indicator
        severity = result['severity']
        colors = {
            'normal': (0, 255, 0),
            'low': (0, 255, 255),
            'medium': (0, 165, 255),
            'high': (0, 0, 255),
            'critical': (128, 0, 128)
        }
        color = colors.get(severity, (255, 255, 255))

        # Add text
        cv2.putText(overlay, f"Severity: {severity.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(overlay, f"Score: {result['anomaly_score']:.4f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return overlay


# =============================================================================
# Training Functions
# =============================================================================

def train_anomaly_detector(
    data_dir: str,
    output_path: str = 'checkpoints/anomaly_detector.pth',
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = 'auto'
):
    """
    Train the anomaly detection autoencoder on healthy cell images.

    Args:
        data_dir: Directory containing healthy cell images
        output_path: Where to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
    """
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from tqdm import tqdm

    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"[*] Training anomaly detector on {device}")

    # Dataset
    class HealthyCellDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.images = []

            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                self.images.extend(list(self.root_dir.rglob(ext)))

            print(f"[+] Found {len(self.images)} images")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    # DataLoader
    dataset = HealthyCellDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model
    model = CellAutoencoder(input_size=128, latent_dim=256)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()

            reconstructed, latent = model(batch)
            loss = F.mse_loss(reconstructed, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'baseline_error': avg_loss,
                'epoch': epoch
            }, output_path)
            print(f"[+] Saved best model (loss: {best_loss:.6f})")

    print(f"\n[+] Training complete! Best loss: {best_loss:.6f}")
    print(f"[+] Model saved to: {output_path}")

    return model, best_loss


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cell Anomaly Detection")
    parser.add_argument('--train', type=str, help="Train on healthy cell images directory")
    parser.add_argument('--analyze', type=str, help="Analyze an image")
    parser.add_argument('--model', type=str, default='checkpoints/anomaly_detector.pth',
                       help="Model path")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    if args.train:
        train_anomaly_detector(
            data_dir=args.train,
            output_path=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    elif args.analyze:
        detector = CellAnomalyDetector(model_path=args.model)
        image = cv2.imread(args.analyze)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = detector.analyze(image)

        print("\n=== Anomaly Detection Result ===")
        print(f"Anomaly Score: {result['anomaly_score']:.6f}")
        print(f"Normalized Score: {result['normalized_score']:.2f}x baseline")
        print(f"Severity: {result['severity'].upper()}")
        print(f"Is Anomaly: {result['is_anomaly']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nPossible Issues: {', '.join(result['possible_issues']) or 'None'}")
        print(f"\nRecommendation: {result['recommendation']}")

        # Save visualization
        vis = detector.create_visualization(image, result)
        output_path = args.analyze.replace('.', '_anomaly.')
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"\n[+] Visualization saved to: {output_path}")

    else:
        parser.print_help()
