"""
Target Format Generation package for ML Pipeline.

This package provides components to generate output in the exact data.json format
required by the business, leveraging enhanced features from the data engineering pipeline.
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"

# Import main components
try:
    from .target_format_generator import TargetFormatGenerator
    from .tension_mapper import TensionMapper
    from .concept_classifier import ConceptClassifier
    
    __all__ = [
        "TargetFormatGenerator",
        "TensionMapper", 
        "ConceptClassifier"
    ]
except ImportError as e:
    print(f"Warning: Some target format components not available: {e}")
    __all__ = []
