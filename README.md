# CellDiagnose-AI

Automated diagnostic tool for brightfield cell microscopy images using deep learning.

## Features

- **Cell Classification**: Identify 13 cell types with 99.64% accuracy
- **Cell Segmentation**: Measure confluency and count cells (90.95% Dice)
- **Anomaly Detection**: Detect contamination, dead cells, and stress
- **Web Interface**: Easy-to-use Streamlit application

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Checkpoints

Download trained model weights and place in `checkpoints/` folder:
- `classifier_best.pth` (133 MB) - EfficientNet-B3
- `segmentation_best.pth` (280 MB) - U-Net
- `anomaly_detector.pth` (38 MB) - Autoencoder

### 3. Run Web Interface

```bash
streamlit run app.py
```

### 4. Or Use Python API

```python
from celldiagnose import CellDiagnoseAI

model = CellDiagnoseAI.load('checkpoints/')
result = model.diagnose('cell_image.png')

print(result.cell_type)        # 'HeLa'
print(result.confluency)       # 45.2
print(result.health_status)    # 'healthy'
```

## Supported Cell Types

| Cell Type | Description |
|-----------|-------------|
| A172 | Human glioblastoma |
| BT474 | Human breast cancer |
| BV2 | Mouse microglia |
| HEK293 | Human embryonic kidney |
| HeLa | Human cervical cancer |
| Hepatocyte | Human liver cells |
| Huh7 | Hepatocellular carcinoma |
| MCF7 | Human breast cancer |
| SHSY5Y | Human neuroblastoma |
| SKOV3 | Human ovarian cancer |
| SkBr3 | Human breast cancer |
| U2OS | Human osteosarcoma |
| U373 | Glioblastoma astrocytoma |

## File Structure

```
CellDiagnose-AI/
├── celldiagnose.py      # Unified inference API
├── app.py               # Streamlit web interface
├── models.py            # Model architectures
├── utils.py             # Image processing utilities
├── config.py            # Configuration
├── anomaly_detection.py # Anomaly detection module
├── train_classifier.py  # Classification training
├── train_segmentation.py# Segmentation training
├── requirements.txt     # Dependencies
├── ARCHITECTURE.md      # System architecture
├── README.md            # This file
└── checkpoints/         # Model weights (not included)
```

## Documentation

- `ARCHITECTURE.md` - System architecture and data flow
- `workhistory.md` - Development history
- `memory.md` - Project status

## License

MIT License
