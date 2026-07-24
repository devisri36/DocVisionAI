import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from configs.config_loader import ConfigManager

def setup_logger(name: str = "docvision") -> logging.Logger:
    """Configures and returns a logger instance with console and file rotating handlers."""
    config = ConfigManager().config
    logger = logging.getLogger(name)
    
    # If logger is already configured, return it
    if logger.handlers:
        return logger

    log_level_str = config.logging.level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(config.logging.format)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # File Rotating Handler
    log_file_path = Path(config.logging.file_path)
    # Ensure logs folder exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if writing to log file fails
        print(f"Warning: Failed to set up file logger handler: {e}", file=sys.stderr)

    return logger
