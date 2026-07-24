import json
from pathlib import Path
from typing import List
from backend.loaders.base import BaseDatasetLoader, DocumentItem
from utils.logger import setup_logger

logger = setup_logger("docvqa_loader")

class DocVQALoader(BaseDatasetLoader):
    """Loader for the DocVQA dataset (Document Visual Question Answering)."""

    def load_data(self) -> List[DocumentItem]:
        split_folder = "validation" if self.split in ["val", "validation"] else "train"
        logger.info(f"Loading DocVQA split '{split_folder}'...")
        
        base_path = self.data_dir / split_folder
        ann_dir = base_path / "annotations"
        img_dir = base_path / "images"

        if not ann_dir.exists():
            logger.warning(f"DocVQA directory not found at {ann_dir}. Returning empty list.")
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

                question = ann_data.get("question", "")
                answers = ann_data.get("answers", [])
                words_list = ann_data.get("words", [])
                raw_bboxes = ann_data.get("bboxes", [])

                bboxes_list = [[float(c) for c in box] for box in raw_bboxes if len(box) == 4]

                if words_list:
                    item = DocumentItem(
                        image_path=img_path,
                        words=words_list,
                        bboxes=bboxes_list,
                        labels=["text"] * len(words_list),
                        metadata={
                            "dataset": "DocVQA",
                            "doc_name": json_file.stem,
                            "question": question,
                            "answers": answers,
                            "doc_id": ann_data.get("docId", "")
                        }
                    )
                    loaded_items.append(item)

            except Exception as e:
                logger.error(f"Error parsing DocVQA annotation file {json_file.name}: {e}")

        logger.info(f"Loaded {len(loaded_items)} documents from DocVQA '{split_folder}' split.")
        return loaded_items
