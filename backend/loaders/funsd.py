import json
from pathlib import Path
from typing import List
from backend.loaders.base import BaseDatasetLoader, DocumentItem
from utils.logger import setup_logger

logger = setup_logger("funsd_loader")

class FUNSDLoader(BaseDatasetLoader):
    """Loader for the FUNSD dataset (Form Understanding in Noisy Scanned Documents)."""

    def load_data(self) -> List[DocumentItem]:
        logger.info(f"Loading FUNSD split '{self.split}'...")
        
        split_folder = "dataset/training_data" if self.split == "train" else "dataset/testing_data"
        base_path = self.data_dir / split_folder
        
        ann_dir = base_path / "annotations"
        img_dir = base_path / "images"

        if not ann_dir.exists():
            base_path = self.data_dir / self.split
            ann_dir = base_path / "annotations"
            img_dir = base_path / "images"

        if not ann_dir.exists():
            logger.warning(f"FUNSD directory not found at {ann_dir}. Returning empty list.")
            return []

        loaded_items = []
        for json_file in ann_dir.glob("*.json"):
            img_path = img_dir / f"{json_file.stem}.png"
            if not img_path.exists():
                img_path = img_dir / f"{json_file.stem}.jpg"
                if not img_path.exists():
                    logger.warning(f"Matching image not found for annotation {json_file.name}. Skipping.")
                    continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)

                words_list = []
                bboxes_list = []
                labels_list = []

                for form_item in ann_data.get("form", []):
                    block_label = form_item.get("label", "other")
                    
                    for word_item in form_item.get("words", []):
                        text = word_item.get("text", "").strip()
                        box = word_item.get("box")
                        
                        if text and box and len(box) == 4:
                            words_list.append(text)
                            bboxes_list.append([float(c) for c in box])
                            labels_list.append(block_label)

                if words_list:
                    item = DocumentItem(
                        image_path=img_path,
                        words=words_list,
                        bboxes=bboxes_list,
                        labels=labels_list,
                        metadata={
                            "dataset": "FUNSD",
                            "doc_name": json_file.stem
                        }
                    )
                    loaded_items.append(item)

            except Exception as e:
                logger.error(f"Error parsing FUNSD annotation file {json_file.name}: {e}")

        logger.info(f"Loaded {len(loaded_items)} documents from FUNSD '{self.split}' split.")
        return loaded_items
