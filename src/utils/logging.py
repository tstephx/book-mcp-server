"""
Logging utilities for MCP server
Follows MCP best practices for logging
"""

import logging
import sys
from typing import Optional

from ..config import Config

def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Setup logging for the server
    
    Args:
        name: Logger name (default: root logger)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name or __name__)
    
    if not Config.ENABLE_LOGGING:
        logger.setLevel(logging.CRITICAL)
        return logger
    
    # Set level based on environment
    level = logging.DEBUG if Config.DEBUG else logging.INFO
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler (stderr for MCP servers)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

# Default logger for the module
logger = setup_logging("book-mcp-server")
