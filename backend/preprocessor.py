import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config_loader import ConfigManager
from utils.logger import setup_logger
from utils.helpers import normalize_bbox

logger = setup_logger("preprocessor")

class DocumentPreprocessor:
    """Document preprocessing pipeline that handles resizing, normalization, augmentations, and bbox alignment."""

    def __init__(self, augment: bool = False):
        self.config = ConfigManager().config
        self.augment = augment and self.config.preprocessing.augmentations.enable
        
        self.target_height = self.config.preprocessing.target_size.height
        self.target_width = self.config.preprocessing.target_size.width
        
        self.mean = self.config.preprocessing.normalization.mean
        self.std = self.config.preprocessing.normalization.std
        
        self._init_transforms()

    def _init_transforms(self):
        """Initializes the Albumentations composition pipeline."""
        transforms = []
        
        # Document-safe augmentations
        if self.augment:
            logger.info("Initializing preprocessor with augmentations enabled.")
            if self.config.preprocessing.augmentations.brightness_contrast:
                transforms.append(A.RandomBrightnessContrast(p=0.5))
            
            rotation_limit = self.config.preprocessing.augmentations.rotation_limit
            if rotation_limit > 0:
                transforms.append(
                    A.ShiftScaleRotate(
                        shift_limit=0.05, 
                        scale_limit=0.05, 
                        rotate_limit=rotation_limit, 
                        border_mode=0, # black borders
                        value=(255, 255, 255), # white fill
                        p=0.4
                    )
                )
        
        # Resize to standard input dims
        transforms.append(A.Resize(height=self.target_height, width=self.target_width))
        
        # Normalize and convert to PyTorch Tensor
        transforms.append(A.Normalize(mean=self.mean, std=self.std))
        transforms.append(ToTensorV2())
        
        # We use pascal_voc format [x_min, y_min, x_max, y_max] for absolute coordinates
        # Albumentations will automatically resize bboxes with the image
        self.pipeline = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_area=1.0,
                min_visibility=0.1
            )
        )

    def _clean_and_clamp_bboxes(
        self, 
        bboxes: List[List[float]], 
        width: int, 
        height: int,
        labels: List[Any],
        words: List[str]
    ) -> Tuple[List[List[float]], List[Any], List[str]]:
        """Cleans and validates bounding boxes to satisfy Albumentations constraints.
        
        Albumentations requires:
        - x_min < x_max and y_min < y_max
        - Coordinates must lie strictly within [0, image_dimension]
        """
        cleaned_bboxes = []
        cleaned_labels = []
        cleaned_words = []
        
        for bbox, label, word in zip(bboxes, labels, words):
            if not bbox or len(bbox) != 4:
                continue
                
            x_min, y_min, x_max, y_max = bbox
            
            # Clamp coordinates to image boundaries
            x_min = max(0.0, min(float(x_min), float(width)))
            y_min = max(0.0, min(float(y_min), float(height)))
            x_max = max(0.0, min(float(x_max), float(width)))
            y_max = max(0.0, min(float(y_max), float(height)))
            
            # Skip zero area boxes
            if x_min >= x_max or y_min >= y_max:
                if x_min == x_max and x_max < width:
                    x_max += 1.0
                if y_min == y_max and y_max < height:
                    y_max += 1.0
                
                if x_min >= x_max or y_min >= y_max:
                    continue
            
            cleaned_bboxes.append([x_min, y_min, x_max, y_max])
            cleaned_labels.append(label)
            cleaned_words.append(word)
            
        return cleaned_bboxes, cleaned_labels, cleaned_words

    def preprocess(
        self, 
        image: Image.Image, 
        bboxes: List[List[float]], 
        words: List[str], 
        labels: List[Any]
    ) -> Dict[str, Any]:
        """Preprocesses a single document: scales/augments image and scales bounding boxes accordingly."""
        img_width, img_height = image.size
        
        # 1. Clean bounding boxes to prevent Albumentations assertion errors
        cleaned_bboxes, cleaned_labels, cleaned_words = self._clean_and_clamp_bboxes(
            bboxes, img_width, img_height, labels, words
        )
        
        # Check if we have no valid bounding boxes after cleaning
        if not cleaned_bboxes:
            cleaned_bboxes = [[0.0, 0.0, 10.0, 10.0]]
            cleaned_labels = [labels[0] if labels else "other"]
            cleaned_words = [words[0] if words else "[PAD]"]

        # 2. Convert image to numpy array
        image_np = np.array(image)
        
        # 3. Apply transformation pipeline
        transformed = self.pipeline(
            image=image_np, 
            bboxes=cleaned_bboxes, 
            class_labels=list(range(len(cleaned_bboxes)))
        )
        
        transformed_image = transformed["image"]
        transformed_bboxes_raw = transformed["bboxes"]
        transformed_indices = transformed["class_labels"]
        
        # 4. Map back transformed elements
        final_bboxes = []
        final_words = []
        final_labels = []
        final_normalized_bboxes = []
        
        for idx, box in zip(transformed_indices, transformed_bboxes_raw):
            final_bboxes.append(list(box))
            final_words.append(cleaned_words[int(idx)])
            final_labels.append(cleaned_labels[int(idx)])
            
            # Scale coordinates to LayoutLM [0, 1000] system
            norm_box = normalize_bbox(box, self.target_width, self.target_height)
            final_normalized_bboxes.append(norm_box)

        return {
            "image": transformed_image,
            "bboxes": final_bboxes,
            "normalized_bboxes": final_normalized_bboxes,
            "words": final_words,
            "labels": final_labels
        }

    @staticmethod
    def train_validation_split(
        items: List[Any], 
        split_ratio: float = 0.8, 
        seed: int = 42
    ) -> Tuple[List[Any], List[Any]]:
        """Splits data items into train and validation sets based on the ratio."""
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(items))
        
        split_idx = int(len(items) * split_ratio)
        train_indices = shuffled_indices[:split_idx]
        val_indices = shuffled_indices[split_idx:]
        
        train_items = [items[idx] for idx in train_indices]
        val_items = [items[idx] for idx in val_indices]
        
        return train_items, val_items
