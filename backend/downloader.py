import os
import json
import zipfile
import io
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw
import numpy as np

from configs.config_loader import ConfigManager
from utils.logger import setup_logger

logger = setup_logger("downloader")

class DatasetDownloader:
    """Downloader class to retrieve and serialize standard vision-document datasets."""

    def __init__(self):
        self.config = ConfigManager().config
        self.datasets_dir = Path(self.config.paths.datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self):
        """Downloads all configured datasets."""
        logger.info("Starting download process for all datasets...")
        self.download_funsd()
        self.download_cord()
        self.download_sroie()
        self.download_docvqa()
        logger.info("All downloads completed.")

    def download_funsd(self) -> Path:
        """Downloads and extracts the FUNSD dataset."""
        funsd_config = self.config.datasets.get("funsd")
        if not funsd_config:
            logger.error("FUNSD configuration is missing.")
            return Path()
            
        target_dir = Path(funsd_config.raw_dir)
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"FUNSD dataset already exists in {target_dir}. Skipping download.")
            return target_dir

        url = funsd_config.download_url
        logger.info(f"Downloading FUNSD dataset from {url}...")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            logger.info("Extracting FUNSD zip file...")
            z = zipfile.ZipFile(io.BytesIO(response.content))
            
            target_dir.mkdir(parents=True, exist_ok=True)
            z.extractall(target_dir)
            logger.info(f"FUNSD successfully extracted to {target_dir}")
        except Exception as e:
            logger.error(f"Failed to download/extract FUNSD: {e}. Generating mock FUNSD data instead.")
            self._create_mock_funsd(target_dir)

        return target_dir

    def download_cord(self) -> Path:
        """Downloads CORD dataset from Hugging Face and saves images + annotations locally."""
        cord_config = self.config.datasets.get("cord")
        if not cord_config:
            logger.error("CORD configuration is missing.")
            return Path()
            
        target_dir = Path(cord_config.raw_dir)
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"CORD dataset already exists in {target_dir}. Skipping download.")
            return target_dir

        hf_dataset_name = cord_config.hf_dataset
        logger.info(f"Attempting to download CORD from Hugging Face dataset: {hf_dataset_name}...")
        try:
            from datasets import load_dataset
            limit = cord_config.sample_limit or 50
            
            for split in ["train", "validation", "test"]:
                logger.info(f"Loading CORD split '{split}' from Hugging Face...")
                dataset = load_dataset(hf_dataset_name, split=split)
                
                split_dir = target_dir / split
                images_dir = split_dir / "images"
                ann_dir = split_dir / "annotations"
                
                images_dir.mkdir(parents=True, exist_ok=True)
                ann_dir.mkdir(parents=True, exist_ok=True)
                
                num_items = min(len(dataset), limit)
                for idx in range(num_items):
                    item = dataset[idx]
                    img = item["image"]
                    gt = item["ground_truth"]
                    
                    filename = f"cord_{idx:04d}"
                    img.save(images_dir / f"{filename}.png")
                    
                    if isinstance(gt, str):
                        gt_dict = json.loads(gt)
                    else:
                        gt_dict = gt
                        
                    with open(ann_dir / f"{filename}.json", "w", encoding="utf-8") as f:
                        json.dump(gt_dict, f, indent=4)
                        
            logger.info(f"CORD successfully saved to {target_dir}")
        except Exception as e:
            logger.error(f"Failed to load CORD from Hugging Face: {e}. Generating mock CORD data instead.")
            self._create_mock_cord(target_dir)

        return target_dir

    def download_sroie(self) -> Path:
        """Downloads SROIE receipt dataset and saves images + text annotations locally."""
        sroie_config = self.config.datasets.get("sroie")
        if not sroie_config:
            logger.error("SROIE configuration is missing.")
            return Path()
            
        target_dir = Path(sroie_config.raw_dir)
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"SROIE dataset already exists in {target_dir}. Skipping download.")
            return target_dir

        hf_dataset_name = sroie_config.hf_dataset
        logger.info(f"Attempting to download SROIE from Hugging Face dataset: {hf_dataset_name}...")
        try:
            from datasets import load_dataset
            limit = sroie_config.sample_limit or 50
            
            for split in ["train", "test"]:
                logger.info(f"Loading SROIE split '{split}' from Hugging Face...")
                dataset = load_dataset(hf_dataset_name, split=split)
                
                split_dir = target_dir / split
                images_dir = split_dir / "images"
                ann_dir = split_dir / "annotations"
                
                images_dir.mkdir(parents=True, exist_ok=True)
                ann_dir.mkdir(parents=True, exist_ok=True)
                
                num_items = min(len(dataset), limit)
                for idx in range(num_items):
                    item = dataset[idx]
                    img = item["image"]
                    
                    filename = f"sroie_{idx:04d}"
                    img.save(images_dir / f"{filename}.png")
                    
                    ann_data = {
                        "words": item.get("words", []),
                        "bboxes": item.get("bboxes", []),
                        "texts": item.get("texts", []),
                        "company": item.get("company", ""),
                        "date": item.get("date", ""),
                        "address": item.get("address", ""),
                        "total": item.get("total", "")
                    }
                    
                    with open(ann_dir / f"{filename}.json", "w", encoding="utf-8") as f:
                        json.dump(ann_data, f, indent=4)
                        
            logger.info(f"SROIE successfully saved to {target_dir}")
        except Exception as e:
            logger.error(f"Failed to load SROIE from Hugging Face: {e}. Generating mock SROIE data instead.")
            self._create_mock_sroie(target_dir)

        return target_dir

    def download_docvqa(self) -> Path:
        """Downloads DocVQA dataset from Hugging Face and saves images + annotations locally."""
        docvqa_config = self.config.datasets.get("docvqa")
        if not docvqa_config:
            logger.error("DocVQA configuration is missing.")
            return Path()
            
        target_dir = Path(docvqa_config.raw_dir)
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"DocVQA dataset already exists in {target_dir}. Skipping download.")
            return target_dir

        hf_dataset_name = docvqa_config.hf_dataset
        logger.info(f"Attempting to download DocVQA from Hugging Face dataset: {hf_dataset_name}...")
        try:
            from datasets import load_dataset
            limit = docvqa_config.sample_limit or 50
            
            for split in ["train", "validation"]:
                logger.info(f"Loading DocVQA split '{split}' from Hugging Face...")
                dataset = load_dataset(hf_dataset_name, split=split)
                
                split_dir = target_dir / split
                images_dir = split_dir / "images"
                ann_dir = split_dir / "annotations"
                
                images_dir.mkdir(parents=True, exist_ok=True)
                ann_dir.mkdir(parents=True, exist_ok=True)
                
                num_items = min(len(dataset), limit)
                for idx in range(num_items):
                    item = dataset[idx]
                    img = item.get("image")
                    if img is None:
                        img_path = item.get("image_path")
                        if img_path and os.path.exists(img_path):
                            img = Image.open(img_path)
                        else:
                            continue
                            
                    filename = f"docvqa_{idx:04d}"
                    img.save(images_dir / f"{filename}.png")
                    
                    ann_data = {
                        "question": item.get("question", ""),
                        "answers": item.get("answers", []),
                        "docId": item.get("docId", ""),
                        "words": item.get("words", []),
                        "bboxes": item.get("bboxes", [])
                    }
                    
                    with open(ann_dir / f"{filename}.json", "w", encoding="utf-8") as f:
                        json.dump(ann_data, f, indent=4)
                        
            logger.info(f"DocVQA successfully saved to {target_dir}")
        except Exception as e:
            logger.error(f"Failed to load DocVQA from Hugging Face: {e}. Generating mock DocVQA data instead.")
            self._create_mock_docvqa(target_dir)

        return target_dir

    # --- MOCK GENERATION HELPERS ---
    def _create_base_mock_image(self, text: str, width: int = 800, height: int = 1000) -> Image.Image:
        """Generates a base canvas image with simulated text rows."""
        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([10, 10, width - 10, height - 10], outline="black", width=2)
        draw.rectangle([20, 20, width - 20, 60], fill="lightgray")
        draw.text((30, 30), f"MOCK: {text}", fill="black")
        return image

    def _create_mock_funsd(self, target_dir: Path):
        """Creates small mock structure matching FUNSD schema."""
        logger.info("Creating mock FUNSD dataset structure...")
        for split in ["dataset/training_data", "dataset/testing_data"]:
            split_dir = target_dir / split
            img_dir = split_dir / "images"
            ann_dir = split_dir / "annotations"
            
            img_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(2):
                img = self._create_base_mock_image(f"FUNSD Document {i}")
                img_filename = f"mock_funsd_{i}.png"
                img.save(img_dir / img_filename)
                
                ann_data = {
                    "form": [
                        {
                            "box": [100, 150, 250, 180],
                            "text": "Invoice Number:",
                            "label": "question",
                            "words": [{"box": [100, 150, 180, 180], "text": "Invoice"}, {"box": [190, 150, 250, 180], "text": "Number:"}],
                            "id": 0,
                            "linking": [[0, 1]]
                        },
                        {
                            "box": [300, 150, 400, 180],
                            "text": "INV-2026-99",
                            "label": "answer",
                            "words": [{"box": [300, 150, 400, 180], "text": "INV-2026-99"}],
                            "id": 1,
                            "linking": [[1, 0]]
                        }
                    ]
                }
                ann_filename = f"mock_funsd_{i}.json"
                with open(ann_dir / ann_filename, "w", encoding="utf-8") as f:
                    json.dump(ann_data, f, indent=4)
        logger.info(f"Mock FUNSD dataset created at {target_dir}")

    def _create_mock_cord(self, target_dir: Path):
        """Creates mock CORD layout."""
        logger.info("Creating mock CORD dataset structure...")
        for split in ["train", "validation", "test"]:
            split_dir = target_dir / split
            img_dir = split_dir / "images"
            ann_dir = split_dir / "annotations"
            
            img_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(2):
                img = self._create_base_mock_image(f"CORD Receipt {i}")
                img_filename = f"cord_{i:04d}.png"
                img.save(img_dir / img_filename)
                
                ann_data = {
                    "menu": [
                        {
                            "name": "HAMBURGER",
                            "price": "5.99",
                            "nm": "HAMBURGER",
                            "price_box": [300, 200, 350, 220],
                            "name_box": [100, 200, 250, 220]
                        }
                    ],
                    "sub_total": {"subtotal_price": "5.99", "box": [300, 300, 350, 320]}
                }
                ann_filename = f"cord_{i:04d}.json"
                with open(ann_dir / ann_filename, "w", encoding="utf-8") as f:
                    json.dump(ann_data, f, indent=4)
        logger.info(f"Mock CORD dataset created at {target_dir}")

    def _create_mock_sroie(self, target_dir: Path):
        """Creates mock SROIE layout."""
        logger.info("Creating mock SROIE dataset structure...")
        for split in ["train", "test"]:
            split_dir = target_dir / split
            img_dir = split_dir / "images"
            ann_dir = split_dir / "annotations"
            
            img_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(2):
                img = self._create_base_mock_image(f"SROIE Receipt {i}")
                img_filename = f"sroie_{i:04d}.png"
                img.save(img_dir / img_filename)
                
                ann_data = {
                    "words": ["BURGER KING", "DATE:", "2026-07-24", "TOTAL:", "$15.50"],
                    "bboxes": [
                        [50, 100, 200, 130],
                        [50, 150, 120, 175],
                        [130, 150, 250, 175],
                        [50, 200, 120, 225],
                        [130, 200, 200, 225]
                    ],
                    "texts": ["BURGER KING", "DATE:", "2026-07-24", "TOTAL:", "$15.50"],
                    "company": "BURGER KING",
                    "date": "2026-07-24",
                    "address": "123 MAIN ST",
                    "total": "$15.50"
                }
                ann_filename = f"sroie_{i:04d}.json"
                with open(ann_dir / ann_filename, "w", encoding="utf-8") as f:
                    json.dump(ann_data, f, indent=4)
        logger.info(f"Mock SROIE dataset created at {target_dir}")

    def _create_mock_docvqa(self, target_dir: Path):
        """Creates mock DocVQA dataset."""
        logger.info("Creating mock DocVQA dataset structure...")
        for split in ["train", "validation"]:
            split_dir = target_dir / split
            img_dir = split_dir / "images"
            ann_dir = split_dir / "annotations"
            
            img_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(2):
                img = self._create_base_mock_image(f"DocVQA Document {i}")
                img_filename = f"docvqa_{i:04d}.png"
                img.save(img_dir / img_filename)
                
                ann_data = {
                    "question": "What is the document title?",
                    "answers": [f"MOCK: DocVQA Document {i}"],
                    "docId": f"doc_{i}",
                    "words": ["MOCK:", "DocVQA", "Document"],
                    "bboxes": [[30, 30, 70, 45], [80, 30, 150, 45], [160, 30, 240, 45]]
                }
                ann_filename = f"docvqa_{i:04d}.json"
                with open(ann_dir / ann_filename, "w", encoding="utf-8") as f:
                    json.dump(ann_data, f, indent=4)
        logger.info(f"Mock DocVQA dataset created at {target_dir}")
