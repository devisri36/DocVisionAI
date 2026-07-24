import unittest
from pathlib import Path
import tempfile
import shutil

from configs.config_loader import ConfigManager
from backend.downloader import DatasetDownloader
from backend.loaders import DatasetLoaderFactory
from backend.loaders.base import PyTorchDocumentDataset
from backend.preprocessor import DocumentPreprocessor

class TestLoaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We will create a temporary directory for datasets config during testing
        cls.temp_dir = tempfile.mkdtemp()
        
        # Override config paths for testing
        cls.config = ConfigManager().config
        
        # Save old raw dirs to restore later
        cls.old_raw_dirs = {}
        for name in ["funsd", "cord", "sroie", "docvqa"]:
            cls.old_raw_dirs[name] = cls.config.datasets[name].raw_dir
            # Override to mock paths inside our temp directory
            cls.config.datasets[name].raw_dir = str(Path(cls.temp_dir) / name)
            
        # Create mock data using DatasetDownloader methods
        downloader = DatasetDownloader()
        downloader._create_mock_funsd(Path(cls.config.datasets["funsd"].raw_dir))
        downloader._create_mock_cord(Path(cls.config.datasets["cord"].raw_dir))
        downloader._create_mock_sroie(Path(cls.config.datasets["sroie"].raw_dir))
        downloader._create_mock_docvqa(Path(cls.config.datasets["docvqa"].raw_dir))

    @classmethod
    def tearDownClass(cls):
        # Restore old config paths
        for name, path in cls.old_raw_dirs.items():
            cls.config.datasets[name].raw_dir = path
            
        # Clean up temp directory
        shutil.rmtree(cls.temp_dir)

    def test_funsd_loader(self):
        loader = DatasetLoaderFactory.get_loader("funsd", split="train")
        items = loader.load_data()
        self.assertTrue(len(items) > 0)
        
        item = items[0]
        self.assertEqual(item.metadata["dataset"], "FUNSD")
        self.assertTrue(len(item.words) > 0)
        self.assertEqual(len(item.words), len(item.bboxes))
        
        # PyTorch Dataset integration
        preprocessor = DocumentPreprocessor(augment=False)
        py_dataset = loader.get_pytorch_dataset()
        py_dataset.preprocessor = preprocessor
        self.assertTrue(len(py_dataset) > 0)
        
        batch = py_dataset[0]
        self.assertIn("image", batch)
        self.assertIn("bboxes", batch)
        self.assertIn("words", batch)

    def test_cord_loader(self):
        loader = DatasetLoaderFactory.get_loader("cord", split="train")
        items = loader.load_data()
        self.assertTrue(len(items) > 0)
        
        item = items[0]
        self.assertEqual(item.metadata["dataset"], "CORD")
        self.assertTrue(len(item.words) > 0)
        self.assertEqual(len(item.words), len(item.labels))

    def test_sroie_loader(self):
        loader = DatasetLoaderFactory.get_loader("sroie", split="train")
        items = loader.load_data()
        self.assertTrue(len(items) > 0)
        
        item = items[0]
        self.assertEqual(item.metadata["dataset"], "SROIE")
        self.assertIn("company", item.labels)

    def test_docvqa_loader(self):
        loader = DatasetLoaderFactory.get_loader("docvqa", split="train")
        items = loader.load_data()
        self.assertTrue(len(items) > 0)
        
        item = items[0]
        self.assertEqual(item.metadata["dataset"], "DocVQA")
        self.assertIn("question", item.metadata)
        self.assertIn("answers", item.metadata)
