from typing import Type
from pathlib import Path

from backend.loaders.base import BaseDatasetLoader, DocumentItem, PyTorchDocumentDataset
from backend.loaders.funsd import FUNSDLoader
from backend.loaders.cord import CORDLoader
from backend.loaders.sroie import SROIELoader
from backend.loaders.docvqa import DocVQALoader
from configs.config_loader import ConfigManager
from utils.logger import setup_logger

logger = setup_logger("loader_factory")

class DatasetLoaderFactory:
    """Factory to instantiate concrete Document loaders dynamically based on dataset configuration names."""
    
    _loaders = {
        "funsd": FUNSDLoader,
        "cord": CORDLoader,
        "sroie": SROIELoader,
        "docvqa": DocVQALoader
    }

    @classmethod
    def get_loader(
        cls, 
        dataset_name: str, 
        split: str = "train", 
        preprocessor=None
    ) -> BaseDatasetLoader:
        """Instantiates and returns the configured dataset loader."""
        dataset_key = dataset_name.lower().strip()
        if dataset_key not in cls._loaders:
            raise ValueError(
                f"Unsupported dataset loader requested: '{dataset_name}'. "
                f"Choose from {list(cls._loaders.keys())}"
            )
            
        config = ConfigManager().config
        dataset_config = config.datasets.get(dataset_key)
        
        if not dataset_config:
            raise ValueError(f"Dataset '{dataset_key}' is not configured in configs/config.yaml")

        loader_class = cls._loaders[dataset_key]
        logger.info(f"Instantiating {loader_class.__name__} for split '{split}' using directory '{dataset_config.raw_dir}'")
        
        return loader_class(
            data_dir=dataset_config.raw_dir, 
            split=split, 
            preprocessor=preprocessor
        )
