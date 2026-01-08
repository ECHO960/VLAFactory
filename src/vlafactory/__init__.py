"""
VLAFactory: A minimal framework for training Vision-Language-Action models.

Built on top of LlamaFactory to extend LLM training capabilities to VLA models.
"""

__version__ = "0.1.0"

from vlafactory.model.vla_model import VLAModel
from vlafactory.data.rlds.dataset import OXERLDSDataset
from vlafactory.trainer.vla_trainer import VLATrainer

__all__ = [
    "VLAModel",
    "OXERLDSDataset",
    "VLATrainer",
]
