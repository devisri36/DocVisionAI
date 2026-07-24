import os
import unittest
from pathlib import Path
from configs.config_loader import ConfigManager

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        # Ensure base config path is resolved
        self.config_path = "configs/config.yaml"
        self.manager = ConfigManager(self.config_path)

    def test_singleton(self):
        manager2 = ConfigManager()
        self.assertIs(self.manager, manager2)

    def test_load_values(self):
        config = self.manager.config
        self.assertIsNotNone(config.project_name)
        self.assertTrue("DocVision" in config.project_name)
        
        # Paths
        self.assertEqual(config.paths.datasets_dir, "datasets")
        
        # Preprocessing
        self.assertEqual(config.preprocessing.target_size.width, 384)
        self.assertEqual(config.preprocessing.target_size.height, 384)

    def test_env_overrides(self):
        # Set temp environment variables
        os.environ["PORT"] = "9999"
        os.environ["LOG_LEVEL"] = "DEBUG"
        
        # Reset instance to force reload
        ConfigManager._instance = None
        new_manager = ConfigManager(self.config_path)
        config = new_manager.config
        
        self.assertEqual(config.api.port, 9999)
        self.assertEqual(config.logging.level, "DEBUG")
        
        # Clean up env
        del os.environ["PORT"]
        del os.environ["LOG_LEVEL"]
        
        # Reset again
        ConfigManager._instance = None
