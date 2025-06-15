#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script to run ML integration tests.

This script ensures the environment is properly configured before running the tests.
"""

import os
import sys
import subprocess
from pathlib import Path

def ensure_package(package_name, version=None):
    """Ensure a Python package is installed."""
    try:
        __import__(package_name)
        print(f"✅ {package_name} is already installed")
    except ImportError:
        version_str = f"=={version}" if version else ""
        print(f"⚠️ Installing {package_name}{version_str}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}{version_str}"])

def main():
    print("Setting up test environment...")
    
    # Ensure critical dependencies are installed
    ensure_package("numpy", "1.24.3")  # Use specific version for compatibility with spaCy
    ensure_package("spacy", "3.7.2")
    
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✅ Added {project_root} to Python path")
    
    # Run the tests
    print("\n🔍 Running ML integration tests...\n")
    os.chdir(str(Path(__file__).parent))
    result = subprocess.call([sys.executable, "-m", "cases.test_ml_integration"])
    
    return result

if __name__ == "__main__":
    sys.exit(main())
