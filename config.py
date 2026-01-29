"""
CellDiagnose-AI: Configuration Management
=========================================
Centralized configuration for the entire application.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Classification
    classification_backbone: str = "efficientnet"  # or "resnet50"
    classification_input_size: int = 224
    classification_pretrained: bool = True

    # Segmentation
    segmentation_model: str = "unet"
    segmentation_input_size: int = 256
    segmentation_channels: int = 3

    # Anomaly detection
    anomaly_threshold: float = 0.3


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # General
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Scheduler
    scheduler: str = "reduce_on_plateau"  # or "cosine", "step"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15

    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")

    # Cell types (loaded from dataset_config.json or defaults)
    cell_types: List[str] = field(default_factory=lambda: [
        'A172', 'BT474', 'BV2', 'HEK293', 'HeLa', 'Hepatocyte', 
        'Huh7', 'MCF7', 'SHSY5Y', 'SKOV3', 'SkBr3', 'U2OS', 'U373'
    ])

    # Paths
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def segmentation_train(self) -> Path:
        return self.processed_dir / "segmentation" / "train"

    @property
    def segmentation_val(self) -> Path:
        return self.processed_dir / "segmentation" / "val"

    @property
    def classification_train(self) -> Path:
        return self.processed_dir / "classification" / "train"

    @property
    def classification_val(self) -> Path:
        return self.processed_dir / "classification" / "val"

    def load_from_json(self):
        """Load cell types from dataset_config.json."""
        config_path = self.processed_dir / "dataset_config.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
                self.cell_types = data.get('cell_types', [])
        return self


@dataclass
class AppConfig:
    """Application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        # Load cell types from processed data
        self.data.load_from_json()


# Global configuration instance
def get_config() -> AppConfig:
    """Get the global configuration."""
    return AppConfig()


# Cell type metadata
CELL_TYPE_INFO = {
    'A172': {
        'name': 'A172',
        'full_name': 'A-172',
        'description': 'Human glioblastoma',
        'organism': 'Human',
        'tissue': 'Brain',
        'disease': 'Glioblastoma'
    },
    'BT474': {
        'name': 'BT474',
        'full_name': 'BT-474',
        'description': 'Human breast cancer',
        'organism': 'Human',
        'tissue': 'Breast',
        'disease': 'Ductal carcinoma'
    },
    'BV2': {
        'name': 'BV2',
        'full_name': 'BV-2',
        'description': 'Mouse microglia',
        'organism': 'Mouse',
        'tissue': 'Brain',
        'disease': 'Immortalized'
    },
    'HeLa': {
        'name': 'HeLa',
        'full_name': 'HeLa',
        'description': 'Human cervical cancer',
        'organism': 'Human',
        'tissue': 'Cervix',
        'disease': 'Adenocarcinoma'
    },
    'Huh7': {
        'name': 'Huh7',
        'full_name': 'Huh-7',
        'description': 'Human hepatocellular carcinoma',
        'organism': 'Human',
        'tissue': 'Liver',
        'disease': 'Hepatocellular carcinoma'
    },
    'MCF7': {
        'name': 'MCF7',
        'full_name': 'MCF-7',
        'description': 'Human breast cancer',
        'organism': 'Human',
        'tissue': 'Breast',
        'disease': 'Adenocarcinoma'
    },
    'SHSY5Y': {
        'name': 'SHSY5Y',
        'full_name': 'SH-SY5Y',
        'description': 'Human neuroblastoma',
        'organism': 'Human',
        'tissue': 'Bone marrow',
        'disease': 'Neuroblastoma'
    },
    'SKOV3': {
        'name': 'SKOV3',
        'full_name': 'SK-OV-3',
        'description': 'Human ovarian cancer',
        'organism': 'Human',
        'tissue': 'Ovary',
        'disease': 'Adenocarcinoma'
    },
    'SkBr3': {
        'name': 'SkBr3',
        'full_name': 'SK-BR-3',
        'description': 'Human breast cancer',
        'organism': 'Human',
        'tissue': 'Breast',
        'disease': 'Adenocarcinoma'
    },
    'U373': {
        'name': 'U373',
        'full_name': 'U-373 MG',
        'description': 'Human glioblastoma astrocytoma',
        'organism': 'Human',
        'tissue': 'Brain',
        'disease': 'Glioblastoma astrocytoma'
    },
    'U2OS': {
        'name': 'U2OS',
        'full_name': 'U-2 OS',
        'description': 'Human osteosarcoma',
        'organism': 'Human',
        'tissue': 'Bone',
        'disease': 'Osteosarcoma'
    },
    'HEK293': {
        'name': 'HEK293',
        'full_name': 'HEK 293',
        'description': 'Human embryonic kidney',
        'organism': 'Human',
        'tissue': 'Kidney',
        'disease': 'Immortalized (adenovirus transformed)'
    },
    'CHO': {
        'name': 'CHO',
        'full_name': 'CHO-K1',
        'description': 'Chinese hamster ovary',
        'organism': 'Chinese hamster',
        'tissue': 'Ovary',
        'disease': 'Immortalized'
    },
    'MDCK': {
        'name': 'MDCK',
        'full_name': 'MDCK',
        'description': 'Madin-Darby canine kidney',
        'organism': 'Dog',
        'tissue': 'Kidney',
        'disease': 'Normal (immortalized)'
    },
}


def get_cell_type_info(cell_type: str) -> Dict:
    """Get metadata for a cell type."""
    return CELL_TYPE_INFO.get(cell_type, {
        'name': cell_type,
        'full_name': cell_type,
        'description': 'Unknown cell type',
        'organism': 'Unknown',
        'tissue': 'Unknown',
        'disease': 'Unknown'
    })
