import os
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if it exists
load_dotenv()

class PathsConfig(BaseModel):
    datasets_dir: str = "datasets"
    logs_dir: str = "logs"
    outputs_dir: str = "outputs"
    models_dir: str = "models"

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file_path: str = "logs/docvision.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class DatasetItemConfig(BaseModel):
    raw_dir: str
    download_url: Optional[str] = None
    hf_dataset: Optional[str] = None
    sample_limit: Optional[int] = None

class PreprocessingSize(BaseModel):
    height: int = 384
    width: int = 384

class NormalizationConfig(BaseModel):
    mean: List[float] = [0.485, 0.456, 0.406]
    std: List[float] = [0.229, 0.224, 0.225]

class AugmentationsConfig(BaseModel):
    enable: bool = True
    rotation_limit: int = 10
    brightness_contrast: bool = True

class PreprocessingConfig(BaseModel):
    target_size: PreprocessingSize = PreprocessingSize()
    normalization: NormalizationConfig = NormalizationConfig()
    augmentations: AugmentationsConfig = AugmentationsConfig()
    train_val_split: float = 0.8

class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class FrontendConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8501

class AppConfig(BaseModel):
    project_name: str = "DocVision AI"
    paths: PathsConfig = PathsConfig()
    logging: LoggingConfig = LoggingConfig()
    datasets: Dict[str, DatasetItemConfig] = {}
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    api: APIConfig = APIConfig()
    frontend: FrontendConfig = FrontendConfig()
    hf_token: Optional[str] = None


class ConfigManager:
    """Manager class for loading, overriding, and validating the application configuration."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config = None
        return cls._instance

    def __init__(self, config_path: str = "configs/config.yaml"):
        if self._config is not None:
            return
        
        self.config_path = Path(config_path)
        self.load_config()

    def load_config(self) -> AppConfig:
        """Loads configuration from YAML file and overrides with environment variables."""
        if not self.config_path.exists():
            # Fallback if config file is not found, use default Pydantic values
            yaml_data = {}
        else:
            with open(self.config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        # Apply basic environment variables overrides to the YAML dictionary before validation
        self._apply_env_overrides(yaml_data)

        # Build configurations
        config = AppConfig(**yaml_data)

        # Hugging Face token from environment variables
        config.hf_token = os.getenv("HF_TOKEN")

        # Create necessary directories listed in paths config
        self._create_directories(config.paths)

        self._config = config
        return self._config

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            self.load_config()
        return self._config

    def _apply_env_overrides(self, data: Dict[str, Any]):
        """Helper to override config parameters from environment variables."""
        # Port & Host
        if os.getenv("HOST"):
            data.setdefault("api", {})["host"] = os.getenv("HOST")
        if os.getenv("PORT"):
            data.setdefault("api", {})["port"] = int(os.getenv("PORT"))
        if os.getenv("STREAMLIT_PORT"):
            data.setdefault("frontend", {})["port"] = int(os.getenv("STREAMLIT_PORT"))

        # Logging
        if os.getenv("LOG_LEVEL"):
            data.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FILE_PATH"):
            data.setdefault("logging", {})["file_path"] = os.getenv("LOG_FILE_PATH")

    def _create_directories(self, paths: PathsConfig):
        """Creates system directories dynamically if they do not exist."""
        for path_str in [paths.datasets_dir, paths.logs_dir, paths.outputs_dir, paths.models_dir]:
            Path(path_str).mkdir(parents=True, exist_ok=True)
