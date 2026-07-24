import json
from pathlib import Path
from typing import List
from backend.loaders.base import BaseDatasetLoader, DocumentItem
from utils.logger import setup_logger

logger = setup_logger("sroie_loader")

class SROIELoader(BaseDatasetLoader):
    """Loader for the SROIE dataset (Scanned Receipts OCR and Information Extraction)."""

    def load_data(self) -> List[DocumentItem]:
        logger.info(f"Loading SROIE split '{self.split}'...")
        
        base_path = self.data_dir / self.split
        ann_dir = base_path / "annotations"
        img_dir = base_path / "images"

        if not ann_dir.exists():
            logger.warning(f"SROIE directory not found at {ann_dir}. Returning empty list.")
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

                raw_words = ann_data.get("words", [])
                raw_bboxes = ann_data.get("bboxes", [])
                
                company = ann_data.get("company", "").strip().lower()
                date = ann_data.get("date", "").strip().lower()
                address = ann_data.get("address", "").strip().lower()
                total = ann_data.get("total", "").strip().lower()

                words_list = []
                bboxes_list = []
                labels_list = []

                for word, box in zip(raw_words, raw_bboxes):
                    if not word or not box or len(box) != 4:
                        continue
                    
                    word_lower = word.strip().lower()
                    
                    label = "other"
                    if company and word_lower in company:
                        label = "company"
                    elif date and word_lower in date:
                        label = "date"
                    elif total and word_lower in total:
                        label = "total"
                    elif address and any(part in address for part in word_lower.split() if len(part) > 2):
                        label = "address"
                        
                    words_list.append(word)
                    bboxes_list.append([float(c) for c in box])
                    labels_list.append(label)

                if words_list:
                    item = DocumentItem(
                        image_path=img_path,
                        words=words_list,
                        bboxes=bboxes_list,
                        labels=labels_list,
                        metadata={
                            "dataset": "SROIE",
                            "doc_name": json_file.stem,
                            "company": ann_data.get("company", ""),
                            "date": ann_data.get("date", ""),
                            "address": ann_data.get("address", ""),
                            "total": ann_data.get("total", "")
                        }
                    )
                    loaded_items.append(item)

            except Exception as e:
                logger.error(f"Error parsing SROIE annotation file {json_file.name}: {e}")

        logger.info(f"Loaded {len(loaded_items)} receipts from SROIE '{self.split}' split.")
        return loaded_items
