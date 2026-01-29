"""
CellDiagnose-AI: Image Processing Utilities
============================================
Helper functions for image preprocessing, segmentation, and analysis.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
from skimage import filters, measure, morphology
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage


class ImagePreprocessor:
    """Handles image loading but preserves original quality for processing."""

    @staticmethod
    def load_image(uploaded_file) -> np.ndarray:
        """Load image from Streamlit uploaded file to numpy array."""
        image = Image.open(uploaded_file)
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range."""
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 0:
            return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return image.astype(np.uint8)

    @staticmethod
    def resize_for_display(image: np.ndarray, max_size: int = 512) -> np.ndarray:
        """Resize image while preserving aspect ratio for display."""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image


class ConfluencyAnalyzer:
    """
    Calculates cell confluency using various segmentation methods.
    Designed as a placeholder for U-Net model integration.
    """

    def __init__(self, method: str = "adaptive_threshold"):
        """
        Initialize analyzer with specified method.

        Args:
            method: One of 'adaptive_threshold', 'otsu', 'canny', 'combined'
        """
        self.method = method
        self._methods = {
            "adaptive_threshold": self._adaptive_threshold_segment,
            "otsu": self._otsu_segment,
            "canny": self._canny_segment,
            "combined": self._combined_segment,
        }

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform confluency analysis on the input image.

        Args:
            image: RGB numpy array

        Returns:
            Dictionary containing:
                - 'mask': Binary segmentation mask
                - 'overlay': Colored overlay for visualization
                - 'confluency_percent': Float percentage of cell coverage
                - 'cell_count_estimate': Estimated number of cells
                - 'method_used': Name of segmentation method
        """
        gray = ImagePreprocessor.to_grayscale(image)

        # Get segmentation mask using selected method
        segment_func = self._methods.get(self.method, self._adaptive_threshold_segment)
        mask = segment_func(gray)

        # Clean up mask with morphological operations
        mask = self._clean_mask(mask)

        # Calculate metrics
        confluency = self._calculate_confluency(mask)
        cell_count = self._estimate_cell_count(mask)

        # Create visualization overlay
        overlay = self._create_overlay(image, mask)

        return {
            'mask': mask,
            'overlay': overlay,
            'confluency_percent': confluency,
            'cell_count_estimate': cell_count,
            'method_used': self.method
        }

    def _adaptive_threshold_segment(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for varying illumination conditions."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Adaptive thresholding
        mask = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=51,
            C=5
        )
        return mask

    def _otsu_segment(self, gray: np.ndarray) -> np.ndarray:
        """Otsu's thresholding for bimodal histograms."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask

    def _canny_segment(self, gray: np.ndarray) -> np.ndarray:
        """Edge-based segmentation using Canny detector."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Fill holes
        mask = ndimage.binary_fill_holes(dilated).astype(np.uint8) * 255
        return mask

    def _combined_segment(self, gray: np.ndarray) -> np.ndarray:
        """Combined approach using multiple methods."""
        adaptive = self._adaptive_threshold_segment(gray)
        otsu = self._otsu_segment(gray)

        # Combine masks (intersection for higher precision)
        combined = cv2.bitwise_and(adaptive, otsu)
        return combined

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean the segmentation mask."""
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Remove small noise
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

        # Fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large, iterations=2)

        # Remove very small objects
        cleaned = morphology.remove_small_objects(
            cleaned.astype(bool),
            min_size=100
        ).astype(np.uint8) * 255

        return cleaned

    def _calculate_confluency(self, mask: np.ndarray) -> float:
        """Calculate percentage of image covered by cells."""
        total_pixels = mask.size
        cell_pixels = np.count_nonzero(mask)
        return round((cell_pixels / total_pixels) * 100, 2)

    def _estimate_cell_count(self, mask: np.ndarray) -> int:
        """Estimate number of cells using connected components."""
        labeled = measure.label(mask)
        regions = measure.regionprops(labeled)

        # Filter by reasonable cell size (adjust based on magnification)
        min_area = 200
        max_area = 50000
        valid_cells = [r for r in regions if min_area < r.area < max_area]

        return len(valid_cells)

    def _create_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create colored overlay showing segmented regions."""
        overlay = image.copy()

        # Create semi-transparent green overlay for cells
        cell_overlay = np.zeros_like(image)
        cell_overlay[:, :, 1] = mask  # Green channel

        # Blend with original
        alpha = 0.4
        overlay = cv2.addWeighted(overlay, 1, cell_overlay, alpha, 0)

        # Add contours for clarity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

        return overlay


class AnomalyDetector:
    """
    Texture-based anomaly detection for contamination screening.
    Placeholder for Autoencoder-based anomaly detection.
    """

    def __init__(self, contamination_threshold: float = 0.3):
        """
        Initialize detector with sensitivity threshold.

        Args:
            contamination_threshold: Score above which image is flagged (0-1)
        """
        self.threshold = contamination_threshold

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image for potential contamination or anomalies.

        Args:
            image: RGB numpy array

        Returns:
            Dictionary containing:
                - 'is_contaminated': Boolean flag
                - 'anomaly_score': Float score (0-1)
                - 'high_freq_energy': High frequency component measure
                - 'texture_irregularity': Texture variance score
                - 'details': Description of findings
        """
        gray = ImagePreprocessor.to_grayscale(image)

        # Calculate various texture metrics
        high_freq = self._analyze_high_frequency(gray)
        texture_score = self._analyze_texture(gray)
        local_variance = self._analyze_local_variance(gray)

        # Combine metrics into anomaly score
        anomaly_score = self._compute_anomaly_score(high_freq, texture_score, local_variance)

        is_contaminated = anomaly_score > self.threshold

        # Generate detailed findings
        details = self._generate_details(high_freq, texture_score, local_variance, is_contaminated)

        return {
            'is_contaminated': is_contaminated,
            'anomaly_score': round(anomaly_score, 3),
            'high_freq_energy': round(high_freq, 3),
            'texture_irregularity': round(texture_score, 3),
            'local_variance': round(local_variance, 3),
            'details': details
        }

    def _analyze_high_frequency(self, gray: np.ndarray) -> float:
        """
        Detect high-frequency noise using FFT.
        High frequency content may indicate bacterial contamination.
        """
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)

        # Calculate magnitude spectrum
        magnitude = np.abs(f_shift)

        # Analyze high frequency region (outer ring)
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4

        # Create mask for high frequency region
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > radius ** 2

        # Calculate ratio of high to total energy
        high_freq_energy = np.sum(magnitude[mask])
        total_energy = np.sum(magnitude)

        if total_energy > 0:
            return high_freq_energy / total_energy
        return 0.0

    def _analyze_texture(self, gray: np.ndarray) -> float:
        """
        Analyze texture using Gray Level Co-occurrence Matrix (GLCM).
        Irregular textures may indicate contamination.
        """
        # Resize for faster processing
        small = cv2.resize(gray, (256, 256))

        # Reduce to fewer gray levels
        small = (small // 16).astype(np.uint8)

        # Calculate GLCM
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        glcm = graycomatrix(small, distances=distances, angles=angles,
                           levels=16, symmetric=True, normed=True)

        # Extract texture properties
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()

        # High contrast + low homogeneity = irregular texture
        irregularity = (contrast / 100) * (1 - homogeneity) * (1 - energy)

        return min(irregularity * 5, 1.0)  # Normalize to 0-1

    def _analyze_local_variance(self, gray: np.ndarray) -> float:
        """
        Analyze local variance to detect unusual patterns.
        Bacterial colonies often create localized high-variance regions.
        """
        # Calculate local variance using a sliding window
        kernel_size = 21

        # Local mean
        local_mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))

        # Local variance
        local_sq = cv2.blur(gray.astype(float) ** 2, (kernel_size, kernel_size))
        local_var = local_sq - local_mean ** 2

        # Calculate coefficient of variation of local variances
        var_std = np.std(local_var)
        var_mean = np.mean(local_var)

        if var_mean > 0:
            cv = var_std / var_mean
            return min(cv / 2, 1.0)  # Normalize to 0-1
        return 0.0

    def _compute_anomaly_score(self, high_freq: float, texture: float,
                               local_var: float) -> float:
        """Combine individual metrics into overall anomaly score."""
        # Weighted combination
        weights = {
            'high_freq': 0.35,
            'texture': 0.35,
            'local_var': 0.30
        }

        score = (
            weights['high_freq'] * high_freq +
            weights['texture'] * texture +
            weights['local_var'] * local_var
        )

        return min(score, 1.0)

    def _generate_details(self, high_freq: float, texture: float,
                          local_var: float, is_contaminated: bool) -> str:
        """Generate human-readable analysis details."""
        findings = []

        if high_freq > 0.4:
            findings.append("High-frequency noise detected (possible microbial presence)")
        if texture > 0.3:
            findings.append("Irregular texture patterns observed")
        if local_var > 0.4:
            findings.append("Unusual local variance detected")

        if is_contaminated:
            return "ALERT: " + "; ".join(findings) if findings else "Anomalous patterns detected"
        elif findings:
            return "Minor observations: " + "; ".join(findings)
        else:
            return "Image appears clean with no significant anomalies"


def create_diagnostic_summary(
    confluency_result: Dict[str, Any],
    classification_result: Dict[str, Any],
    anomaly_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine all analysis results into a structured diagnostic summary.

    Args:
        confluency_result: Output from ConfluencyAnalyzer
        classification_result: Output from CellClassifier model
        anomaly_result: Output from AnomalyDetector

    Returns:
        Comprehensive diagnostic report dictionary
    """
    # Determine overall health status
    health_factors = []

    if confluency_result['confluency_percent'] < 20:
        health_factors.append("Low confluency")
    elif confluency_result['confluency_percent'] > 95:
        health_factors.append("Overly confluent")

    if anomaly_result['is_contaminated']:
        health_factors.append("Potential contamination")

    if classification_result.get('health_status') == 'Unhealthy':
        health_factors.append("Unhealthy cell morphology")

    overall_status = "Healthy" if len(health_factors) == 0 else "Needs Attention"

    return {
        'confluency': {
            'percentage': confluency_result['confluency_percent'],
            'cell_count': confluency_result['cell_count_estimate'],
            'method': confluency_result['method_used']
        },
        'classification': {
            'cell_type': classification_result.get('cell_type', 'Unknown'),
            'cell_type_confidence': classification_result.get('cell_type_confidence', 0.0),
            'health_status': classification_result.get('health_status', 'Unknown'),
            'health_confidence': classification_result.get('health_confidence', 0.0)
        },
        'contamination': {
            'is_contaminated': anomaly_result['is_contaminated'],
            'score': anomaly_result['anomaly_score'],
            'details': anomaly_result['details']
        },
        'overall': {
            'status': overall_status,
            'concerns': health_factors,
            'recommendation': _get_recommendation(overall_status, health_factors)
        }
    }


def _get_recommendation(status: str, concerns: list) -> str:
    """Generate actionable recommendation based on findings."""
    if status == "Healthy":
        return "Culture appears healthy. Continue standard monitoring protocol."

    recommendations = []
    for concern in concerns:
        if "contamination" in concern.lower():
            recommendations.append("Perform sterility check and consider discarding if confirmed")
        if "confluency" in concern.lower():
            recommendations.append("Consider passaging cells or adjusting seeding density")
        if "unhealthy" in concern.lower():
            recommendations.append("Review culture conditions and media freshness")

    return "; ".join(recommendations) if recommendations else "Review culture conditions"
