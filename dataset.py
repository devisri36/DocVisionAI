import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
from PIL import Image

from backend.loaders import DatasetLoaderFactory
from backend.loaders.base import DocumentItem
from backend.preprocessor import DocumentPreprocessor
from utils.logger import setup_logger

logger = setup_logger("dataset_florence")

class FlorenceDocumentDataset(Dataset):
    """PyTorch Dataset that maps unified DocumentItems to Florence-2 sequence-to-sequence tokens."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        processor: Any = None,
        preprocessor: DocumentPreprocessor = None
    ):
        """
        Args:
            dataset_name: Name of dataset ("funsd", "cord", "sroie", "docvqa").
            split: Data split ("train", "val", "test").
            processor: Hugging Face Florence-2 Processor instance.
            preprocessor: Optional custom DocumentPreprocessor from Phase 1.
        """
        self.dataset_name = dataset_name.lower().strip()
        self.split = split
        self.processor = processor
        self.preprocessor = preprocessor
        
        # Instantiate base dataset loader from factory
        self.loader = DatasetLoaderFactory.get_loader(
            dataset_name=self.dataset_name,
            split=self.split
        )
        
        # Load raw items
        self.raw_items: List[DocumentItem] = self.loader.load_data()
        logger.info(f"Initialized FlorenceDataset for {self.dataset_name} ({split}): loaded {len(self.raw_items)} items.")

    def __len__(self) -> int:
        return len(self.raw_items)

    def _generate_florence_target(self, item: DocumentItem) -> Tuple[str, str]:
        """Generates Florence-2 compatible prompt and target text based on dataset task."""
        
        if self.dataset_name == "docvqa":
            # Document QA Task
            question = item.metadata.get("question", "")
            answers = item.metadata.get("answers", [])
            answer = answers[0] if answers else ""
            
            prompt = f"<DocVQA> Question: {question}"
            target = answer
            
        elif self.dataset_name == "sroie":
            # Key Information Extraction Task
            company = item.metadata.get("company", "")
            date = item.metadata.get("date", "")
            address = item.metadata.get("address", "")
            total = item.metadata.get("total", "")
            
            prompt = "<KIE>"
            target = f"company: {company} | date: {date} | address: {address} | total: {total}"
            
        elif self.dataset_name in ["funsd", "cord"]:
            # Visual Entity Recognition / OCR with coordinates
            # Target output: word1 <loc_y1><loc_x1><loc_y2><loc_x2> (label1) | ...
            prompt = "<OCR_WITH_REGION>"
            
            parts = []
            for word, box, label in zip(item.words, item.bboxes, item.labels):
                # Florence-2 box formatting: values are scaled 0-1000 and mapped as coordinates
                # Since preprocessor/helpers normalized_bbox is already 0-1000, we check if they are normalized.
                # Box coordinates should be integers
                x0, y0, x1, y1 = [int(c) for c in box]
                
                # Florence-2 location token standard: y1, x1, y2, x2 scaled to 1000
                # In standard Florence-2 location format, coordinates are y1, x1, y2, x2
                loc_str = f"<loc_{y0}><loc_{x0}><loc_{y1}><loc_{x1}>"
                parts.append(f"{word} {loc_str} ({label})")
                
            target = " | ".join(parts)
            
        else:
            # Generic OCR Task fallback
            prompt = "<OCR>"
            target = " ".join(item.words)

        return prompt, target

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.raw_items[idx]
        
        # Load image safely
        try:
            image = Image.open(item.image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {item.image_path}: {e}")
            image = Image.new("RGB", (384, 384), color="white")

        words = item.words
        bboxes = item.bboxes
        labels = item.labels

        # Apply Phase 1 preprocessor if present (handles resizing and augmentations)
        if self.preprocessor:
            processed = self.preprocessor.preprocess(
                image=image,
                bboxes=bboxes,
                words=words,
                labels=labels
            )
            # Recreate PIL image from the preprocessed canvas to feed into Florence processor
            # Since the preprocessor normalizes the image, we can just use the resized PIL representation
            # to let the Florence-2 processor do its own model-specific image scaling and Normalization.
            image = Image.fromarray(
                ((processed["image"].permute(1, 2, 0).numpy() * 
                  self.preprocessor.std + self.preprocessor.mean) * 255).astype("uint8")
            )
            # Update coordinate boxes if they were transformed during resize/augment
            item.bboxes = processed["bboxes"]

        # Generate sequence prompt and text target
        prompt, target_text = self._generate_florence_target(item)

        # Process through Florence-2 processor (if available)
        if self.processor is not None:
            # Tokenize prompt and process image
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            # Squeeze batch dimension for collator batching
            pixel_values = inputs["pixel_values"].squeeze(0)
            input_ids = inputs["input_ids"].squeeze(0)
            
            # Tokenize target text for loss calculation
            labels = self.processor.tokenizer(
                text=target_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024
            ).input_ids.squeeze(0)
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "labels": labels,
                "prompt": prompt,
                "target_text": target_text,
                "image_path": str(item.image_path)
            }
        else:
            # Fallback if no processor is provided (e.g. during standalone testing)
            return {
                "image": image,
                "prompt": prompt,
                "target_text": target_text,
                "image_path": str(item.image_path)
            }


class FlorenceDataCollator:
    """Collator that pads tokenized input_ids and labels for sequence-to-sequence training."""

    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        
        # Pad input_ids
        input_ids = [item["input_ids"] for item in batch]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        
        # Pad attention masks (1 for real tokens, 0 for pad)
        attention_masks = []
        for ids in input_ids:
            mask = torch.ones_like(ids)
            attention_masks.append(mask)
            
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0
        )
        
        # Pad labels (seq2seq targets)
        labels = [item["labels"] for item in batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.label_pad_token_id
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
        }
