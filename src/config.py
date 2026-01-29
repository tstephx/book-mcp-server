"""
Configuration management for Book MCP Server
Follows MCP best practices for environment-based configuration
"""

import os
from pathlib import Path
from typing import Optional

class Config:
    """Server configuration with environment variable support"""
    
    # Server metadata
    SERVER_NAME: str = os.getenv("MCP_SERVER_NAME", "book-library")
    SERVER_VERSION: str = os.getenv("MCP_SERVER_VERSION", "1.0.0")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Paths - default to book-ingestion-python project
    _DEFAULT_DB = str(Path(__file__).parent.parent.parent / "book-ingestion-python" / "data" / "library.db")
    _DEFAULT_BOOKS = str(Path(__file__).parent.parent.parent / "book-ingestion-python" / "data" / "books")
    
    DB_PATH: Path = Path(os.getenv("BOOK_DB_PATH", _DEFAULT_DB))
    BOOKS_DIR: Path = Path(os.getenv("BOOKS_DIR", _DEFAULT_BOOKS))
    
    # Search limits
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    MAX_CHAPTER_SIZE: int = int(os.getenv("MAX_CHAPTER_SIZE", "100000"))  # 100KB
    
    # Features
    ENABLE_LOGGING: bool = os.getenv("ENABLE_LOGGING", "true").lower() == "true"
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "false").lower() == "true"

    # Cache settings
    CACHE_CHAPTER_TTL: int = int(os.getenv("CACHE_CHAPTER_TTL", "3600"))  # 1 hour
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []
        
        if not cls.DB_PATH.exists():
            errors.append(f"Database not found: {cls.DB_PATH}")
        
        if not cls.BOOKS_DIR.exists():
            errors.append(f"Books directory not found: {cls.BOOKS_DIR}")
        
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    @classmethod
    def display(cls) -> str:
        """Display configuration (for debugging)"""
        return f"""
Book MCP Server Configuration
=============================
Server: {cls.SERVER_NAME} v{cls.SERVER_VERSION}
Environment: {cls.ENVIRONMENT}
Debug: {cls.DEBUG}

Paths:
  Database: {cls.DB_PATH}
  Books: {cls.BOOKS_DIR}

Limits:
  Max search results: {cls.MAX_SEARCH_RESULTS}
  Max chapter size: {cls.MAX_CHAPTER_SIZE} bytes

Features:
  Logging: {cls.ENABLE_LOGGING}
  Caching: {cls.ENABLE_CACHING}
  Cache Chapter TTL: {cls.CACHE_CHAPTER_TTL}s
=============================
"""
