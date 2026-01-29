#!/usr/bin/env python3
"""
Convenience wrapper to run the server
Use this or run directly: python -m src.server
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from src.server import main
    main()
