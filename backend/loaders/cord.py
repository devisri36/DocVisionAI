import json
from pathlib import Path
from typing import List, Dict, Any
from backend.loaders.base import BaseDatasetLoader, DocumentItem
from utils.logger import setup_logger

logger = setup_logger("cord_loader")

class CORDLoader(BaseDatasetLoader):
    """Loader for the CORD dataset (Consolidated Receipt Dataset for Post-OCR Parsing)."""

    def load_data(self) -> List[DocumentItem]:
        logger.info(f"Loading CORD split '{self.split}'...")
        
        base_path = self.data_dir / self.split
        ann_dir = base_path / "annotations"
        img_dir = base_path / "images"

        if not ann_dir.exists():
            logger.warning(f"CORD directory not found at {ann_dir}. Returning empty list.")
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

                self._parse_dict(ann_data, words_list, bboxes_list, labels_list)

                if words_list:
                    item = DocumentItem(
                        image_path=img_path,
                        words=words_list,
                        bboxes=bboxes_list,
                        labels=labels_list,
                        metadata={
                            "dataset": "CORD",
                            "doc_name": json_file.stem
                        }
                    )
                    loaded_items.append(item)

            except Exception as e:
                logger.error(f"Error parsing CORD annotation file {json_file.name}: {e}")

        logger.info(f"Loaded {len(loaded_items)} documents from CORD '{self.split}' split.")
        return loaded_items

    def _parse_dict(self, data: Dict[str, Any], words: List[str], bboxes: List[List[float]], labels: List[str]):
        """Recursively parses receipt items, looking for keys containing bounding boxes and texts."""
        if "valid_line" in data:
            for line in data["valid_line"]:
                self._parse_dict(line, words, bboxes, labels)
            return

        for key, value in data.items():
            if key.endswith("_box") or key == "box":
                label = key.replace("_box", "")
                box = value
                
                sibling_text = ""
                if label == "price" and "price" in data:
                    sibling_text = str(data["price"])
                elif label == "name" and "name" in data:
                    sibling_text = str(data["name"])
                elif label == "subtotal" and "subtotal_price" in data:
                    sibling_text = str(data["subtotal_price"])
                elif "text" in data:
                    sibling_text = str(data["text"])
                else:
                    sibling_text = key.upper()
                
                if box and isinstance(box, list) and len(box) == 4:
                    words.append(sibling_text)
                    bboxes.append([float(c) for c in box])
                    labels.append(label)

            if isinstance(value, dict):
                self._parse_dict(value, words, bboxes, labels)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (dict, list)):
                        self._parse_dict(item, words, bboxes, labels)
