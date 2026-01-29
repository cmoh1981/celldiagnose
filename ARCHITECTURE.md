# CellDiagnose-AI Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CellDiagnose-AI                                   │
│                   Unified Cell Microscopy Diagnostics                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Image Input: .jpg, .png, .tif, .tiff (brightfield microscopy)       │  │
│  │  Sources: File path, NumPy array, PIL Image, Streamlit upload        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING LAYER                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Classification  │  │  Segmentation   │  │ Anomaly Detect  │             │
│  │    300×300       │  │    512×512      │  │    128×128      │             │
│  │  ImageNet Norm   │  │  ImageNet Norm  │  │   [0,1] Norm    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL LAYER                                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CLASSIFICATION MODEL                              │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  EfficientNet-B3 (pretrained ImageNet)                      │   │    │
│  │  │  Parameters: 11.49M                                          │   │    │
│  │  │                                                              │   │    │
│  │  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │   │    │
│  │  │  │   Backbone   │ -> │  Global Avg  │ -> │  Classifier  │  │   │    │
│  │  │  │ EfficientNet │    │    Pool      │    │    Head      │  │   │    │
│  │  │  │   Blocks     │    │  (1536 dim)  │    │ 1536->512->13│  │   │    │
│  │  │  └──────────────┘    └──────────────┘    └──────────────┘  │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │  Output: 13-class probabilities (A172, BT474, BV2, HEK293, HeLa,   │    │
│  │          Hepatocyte, Huh7, MCF7, SHSY5Y, SKOV3, SkBr3, U2OS, U373) │    │
│  │  Accuracy: 99.64% | Balanced Acc: 99.92%                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     SEGMENTATION MODEL                               │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  U-Net + ResNet-34 Encoder (segmentation_models_pytorch)    │   │    │
│  │  │  Parameters: 24.4M                                           │   │    │
│  │  │                                                              │   │    │
│  │  │        ┌─────────────────────────────────────────┐          │   │    │
│  │  │        │              U-Net Architecture          │          │   │    │
│  │  │        │                                          │          │   │    │
│  │  │   Input│    ┌────┐  ┌────┐  ┌────┐  ┌────┐      │Output    │   │    │
│  │  │  512×512│ -> │enc1│->│enc2│->│enc3│->│enc4│->    │512×512   │   │    │
│  │  │   RGB  │    └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘      │ Mask     │   │    │
│  │  │        │      │ Skip  │ Skip  │ Skip  │ Bottleneck         │   │    │
│  │  │        │      ▼ Conn  ▼ Conn  ▼ Conn  ▼          │          │   │    │
│  │  │        │    ┌────┐  ┌────┐  ┌────┐  ┌────┐      │          │   │    │
│  │  │        │ <- │dec1│<-│dec2│<-│dec3│<-│dec4│<-    │          │   │    │
│  │  │        │    └────┘  └────┘  └────┘  └────┘      │          │   │    │
│  │  │        └─────────────────────────────────────────┘          │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │  Output: Binary segmentation mask (0=background, 255=cell)         │    │
│  │  Performance: Dice=0.9095 | IoU=0.834                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   ANOMALY DETECTION MODEL                            │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  Convolutional Autoencoder                                   │   │    │
│  │  │  Strategy: Train on healthy cells, detect high recon error  │   │    │
│  │  │                                                              │   │    │
│  │  │   ┌───────────────────────────────────────────────────────┐ │   │    │
│  │  │   │                                                        │ │   │    │
│  │  │   │  ENCODER                    DECODER                    │ │   │    │
│  │  │   │  128×128×3                  128×128×3                  │ │   │    │
│  │  │   │     │                           ▲                      │ │   │    │
│  │  │   │  Conv 32 (64×64)         Deconv 32 (64×64)            │ │   │    │
│  │  │   │     │                           ▲                      │ │   │    │
│  │  │   │  Conv 64 (32×32)         Deconv 64 (32×32)            │ │   │    │
│  │  │   │     │                           ▲                      │ │   │    │
│  │  │   │  Conv 128 (16×16)        Deconv 128 (16×16)           │ │   │    │
│  │  │   │     │                           ▲                      │ │   │    │
│  │  │   │  Conv 256 (8×8)          Deconv 256 (8×8)             │ │   │    │
│  │  │   │     │                           ▲                      │ │   │    │
│  │  │   │  Conv 512 (4×4)          Deconv 512 (4×4)             │ │   │    │
│  │  │   │     │                           ▲                      │ │   │    │
│  │  │   │     └──> FC 256 ──> FC ─────────┘                      │ │   │    │
│  │  │   │          (Latent)                                      │ │   │    │
│  │  │   └───────────────────────────────────────────────────────┘ │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │  Output: Reconstruction error (anomaly score), severity level       │    │
│  │  Detects: Bacteria, dead cells, stress, debris, mycoplasma         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AGGREGATION LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     DiagnosisResult                                  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │    │
│  │  │ cell_type   │  │ confluency  │  │ health_     │                 │    │
│  │  │ confidence  │  │ cell_count  │  │ status      │                 │    │
│  │  │ probs{}     │  │ mask        │  │ anomaly_    │                 │    │
│  │  │             │  │ overlay     │  │ score       │                 │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │    │
│  │                          │                                          │    │
│  │                          ▼                                          │    │
│  │              ┌─────────────────────────────────┐                   │    │
│  │              │    Overall Health Assessment    │                   │    │
│  │              │  - healthy / needs_attention /  │                   │    │
│  │              │    unhealthy                    │                   │    │
│  │              │  - Recommendation text          │                   │    │
│  │              └─────────────────────────────────┘                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   JSON Report   │  │  Visualization  │  │  Streamlit UI   │             │
│  │  .to_json()     │  │  .visualize()   │  │    app.py       │             │
│  │  .to_dict()     │  │  .save_mask()   │  │                 │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌──────────────┐
│   Input      │
│   Image      │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    CellDiagnoseAI.diagnose()                 │
│                                                              │
│  1. Load & Validate Image                                    │
│     └─> Convert to RGB numpy array                          │
│                                                              │
│  2. Parallel Model Inference                                 │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │ │
│     │  │ classify │  │ segment  │  │ detect_  │          │ │
│     │  │   ()     │  │   ()     │  │ anomalies│          │ │
│     │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │ │
│     │       │             │             │                 │ │
│     │       ▼             ▼             ▼                 │ │
│     │  cell_type     mask, count    anomaly_score        │ │
│     │  confidence    confluency     severity             │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                              │
│  3. Aggregate Results                                        │
│     └─> Determine overall health status                     │
│     └─> Generate recommendations                            │
│                                                              │
│  4. Return DiagnosisResult                                   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    DiagnosisResult                           │
│                                                              │
│  Attributes:                                                 │
│  ├── cell_type: str           ("HeLa")                      │
│  ├── cell_type_confidence: float  (0.987)                   │
│  ├── confluency: float        (45.2%)                       │
│  ├── cell_count: int          (127)                         │
│  ├── segmentation_mask: ndarray                             │
│  ├── health_status: str       ("healthy")                   │
│  ├── anomaly_score: float     (0.023)                       │
│  ├── anomaly_severity: str    ("normal")                    │
│  ├── possible_issues: List[str]                             │
│  └── recommendation: str                                    │
│                                                              │
│  Methods:                                                    │
│  ├── .to_dict()          -> Dict                            │
│  ├── .to_json(path)      -> str / file                      │
│  ├── .visualize()        -> ndarray (overlay image)         │
│  ├── .save_mask(path)    -> file                            │
│  └── .save_visualization(path) -> file                      │
└──────────────────────────────────────────────────────────────┘
```

## File Structure

```
CellDiagnose-AI/
│
├── celldiagnose.py          # UNIFIED INFERENCE PIPELINE (NEW)
│   ├── CellDiagnoseAI       # Main class
│   ├── DiagnosisResult      # Result dataclass
│   ├── ClassificationResult # Classification output
│   ├── SegmentationResult   # Segmentation output
│   └── AnomalyResult        # Anomaly detection output
│
├── app.py                   # Streamlit Web Interface
│   └── Uses celldiagnose.py for backend
│
├── checkpoints/             # Trained Models
│   ├── classifier_best.pth      # EfficientNet-B3 (133 MB)
│   ├── segmentation_best.pth    # U-Net+ResNet34 (280 MB)
│   └── anomaly_detector.pth     # Autoencoder (38 MB)
│
├── Training Scripts
│   ├── train_classifier.py      # Classification training
│   ├── train_segmentation.py    # Segmentation training
│   └── anomaly_detection.py     # Anomaly detector training
│
├── Utilities
│   ├── models.py                # Model architectures (legacy)
│   ├── utils.py                 # Image processing utilities
│   └── config.py                # Configuration management
│
└── data/
    └── processed/
        ├── classification/      # 7,412 images (13 classes)
        │   ├── train/          # 5,159 images
        │   ├── val/            # 1,156 images
        │   └── test/           # 1,097 images
        └── segmentation/        # 3,440 image+mask pairs
            ├── train/
            └── val/
```

## Usage Examples

### Python API

```python
from celldiagnose import CellDiagnoseAI

# Load model
model = CellDiagnoseAI.load('checkpoints/')

# Complete diagnosis
result = model.diagnose('cell_image.png')
print(f"Cell type: {result.cell_type} ({result.cell_type_confidence:.1%})")
print(f"Confluency: {result.confluency:.1f}%")
print(f"Health: {result.health_status}")

# Export results
result.to_json('report.json')
result.save_visualization('diagnosis.png')

# Individual analyses
classification = model.classify('image.png')
segmentation = model.segment('image.png')
anomaly = model.detect_anomalies('image.png')
```

### Command Line

```bash
# Basic diagnosis
python celldiagnose.py cell_image.png

# With outputs
python celldiagnose.py cell_image.png -o report.json -v visualization.png

# Specify checkpoint directory
python celldiagnose.py cell_image.png -c /path/to/checkpoints/
```

### Streamlit Web App

```bash
streamlit run app.py
```

## Model Performance Summary

| Model | Architecture | Parameters | Performance | Checkpoint Size |
|-------|-------------|------------|-------------|-----------------|
| Classification | EfficientNet-B3 | 11.49M | 99.64% acc, 99.92% balanced | 133 MB |
| Segmentation | U-Net + ResNet-34 | 24.4M | 90.95% Dice, 83.4% IoU | 280 MB |
| Anomaly Detection | Conv Autoencoder | ~5M | MSE-based reconstruction | 38 MB |

## Supported Cell Types (13 classes)

| Cell Type | Description | Source |
|-----------|-------------|--------|
| A172 | Human glioblastoma | BBBC |
| BT474 | Human breast cancer | LIVECell |
| BV2 | Mouse microglia | IDR |
| HEK293 | Human embryonic kidney | Various |
| HeLa | Human cervical cancer | LIVECell |
| Hepatocyte | Human liver cells | Custom |
| Huh7 | Hepatocellular carcinoma | Various |
| MCF7 | Human breast cancer | LIVECell |
| SHSY5Y | Human neuroblastoma | LIVECell |
| SKOV3 | Human ovarian cancer | LIVECell |
| SkBr3 | Human breast cancer | LIVECell |
| U2OS | Human osteosarcoma | BBBC |
| U373 | Glioblastoma astrocytoma | Various |

## System Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **GPU**: NVIDIA with CUDA (recommended), CPU supported
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: ~500MB for models
