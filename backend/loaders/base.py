from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.logger import setup_logger

logger = setup_logger("loaders")

class DocumentItem:
    """A standard class representing a unified document sample in the DocVision system."""
    def __init__(
        self,
        image_path: Path,
        words: List[str],
        bboxes: List[List[float]],
        labels: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.image_path = image_path
        self.words = words
        self.bboxes = bboxes  # Raw absolute bounding boxes [x0, y0, x1, y1]
        self.labels = labels if labels is not None else []
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "words": self.words,
            "bboxes": self.bboxes,
            "labels": self.labels,
            "metadata": self.metadata
        }


class PyTorchDocumentDataset(Dataset):
    """PyTorch Dataset wrapper for unified DocumentItem collections, integrated with preprocessors."""
    
    def __init__(self, items: List[DocumentItem], preprocessor=None):
        self.items = items
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        
        try:
            image = Image.open(item.image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {item.image_path}: {e}")
            image = Image.new("RGB", (384, 384), color="white")

        words = item.words
        bboxes = item.bboxes
        labels = item.labels

        if self.preprocessor:
            processed = self.preprocessor.preprocess(
                image=image, 
                bboxes=bboxes, 
                words=words, 
                labels=labels
            )
            image_tensor = processed["image"]
            transformed_bboxes = processed["bboxes"]
            transformed_words = processed["words"]
            transformed_labels = processed["labels"]
        else:
            image_tensor = image
            transformed_bboxes = bboxes
            transformed_words = words
            transformed_labels = labels

        return {
            "image": image_tensor,
            "bboxes": transformed_bboxes,
            "words": transformed_words,
            "labels": transformed_labels,
            "metadata": item.metadata,
            "image_path": str(item.image_path)
        }


class BaseDatasetLoader(ABC):
    """Abstract Base Class for specific vision-document dataset loaders."""
    
    def __init__(self, data_dir: str, split: str = "train", preprocessor=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.preprocessor = preprocessor
        self.items: List[DocumentItem] = []
        
    @abstractmethod
    def load_data(self) -> List[DocumentItem]:
        """Parses annotations and links images to build list of DocumentItems."""
        pass

    def get_pytorch_dataset(self) -> PyTorchDocumentDataset:
        """Returns initialized PyTorch dataset containing pre-loaded items."""
        if not self.items:
            self.items = self.load_data()
        return PyTorchDocumentDataset(self.items, preprocessor=self.preprocessor)
