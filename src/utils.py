#!/usr/bin/env python3
"""Utility functions for research projects."""

import os
import shutil
import sys
from pathlib import Path
from typing import Union
from dotenv import load_dotenv


def init_directory(directory: Union[str, Path], overwrite: bool = False) -> Path:
    """
    Initialize a directory with safety checks for overwriting.
    
    This is a generic tool for safely creating/overwriting directories. It uses the
    DATA_DIR environment variable to specify a safe prefix - only directories 
    under this prefix can be overwritten. This prevents accidental deletion of 
    important system directories.
    
    Args:
        directory: Path to directory (str or Path object)
        overwrite: Whether to overwrite existing directory
    
    Returns:
        Path object of the created directory
    
    Raises:
        SystemExit: If directory exists without overwrite, or safety checks fail
    """
    load_dotenv()
    
    directory = Path(directory)
    
    if directory.exists():
        if overwrite:
            # Get DATA_DIR from environment (loaded from .env)
            safe_prefix = os.environ.get('DATA_DIR')
            
            if not safe_prefix:
                print(f"Error: DATA_DIR not set in .env!")
                print(f"Cannot use --overwrite without DATA_DIR for safety.")
                print("Set DATA_DIR in .env file to specify where overwriting is allowed.")
                sys.exit(1)
            
            # Convert safe_prefix to absolute path for comparison
            safe_prefix = Path(safe_prefix).resolve()
            
            # Get absolute path of directory
            dir_absolute = directory.resolve()
            
            # Check if the absolute path starts with safe prefix
            if not str(dir_absolute).startswith(str(safe_prefix)):
                print(f"Error: Cannot overwrite {dir_absolute}")
                print(f"Directory must start with DATA_DIR: {safe_prefix}")
                print("This safety check prevents accidental deletion of important directories.")
                sys.exit(1)
            
            # Safe to remove
            print(f"Removing existing directory: {dir_absolute}")
            shutil.rmtree(dir_absolute)
            print("Directory removed successfully.")
        else:
            print(f"Error: Directory {directory} already exists!")
            print("Use --overwrite to remove it, or choose a different path.")
            sys.exit(1)
    
    # Create directory
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Created directory: {directory.resolve()}")
    return directory


# ============================================================================
# Other reusable utilities for the research
# ============================================================================
# Add stateless utility functions below that are expected to be used 
# repetitively throughout the research project