import unittest
import logging
from pathlib import Path
from utils.logger import setup_logger

class TestLogger(unittest.TestCase):
    def test_logger_setup(self):
        logger = setup_logger("test_docvision")
        self.assertIsInstance(logger, logging.Logger)
        self.assertTrue(len(logger.handlers) >= 1)
        
        # Test logging doesn't crash
        with self.assertLogs("test_docvision", level="INFO") as cm:
            logger.info("Logging configuration test message.")
        
        self.assertEqual(cm.output, ["INFO:test_docvision:Logging configuration test message."])
        
        # Verify log file handler exists and directory is present
        from configs.config_loader import ConfigManager
        config = ConfigManager().config
        log_file = Path(config.logging.file_path)
        self.assertTrue(log_file.parent.exists())
