"""
ML Models Package for Hybrid Classification Pipeline

This package contains machine learning model implementations for enhancing
the classification pipeline components that are underperforming.

Components:
- tension_detection: Models for improving tension detection from 33% to 75-90% accuracy
- thematic_classification: Models for improving thematic classification to 85-95% accuracy
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"

# Model categories
TENSION_DETECTION_MODELS = [
    "RandomForestTensionDetector",
    "XGBoostTensionDetector", 
    "SVMTensionDetector",
    "EnsembleTensionDetector"
]

THEMATIC_CLASSIFICATION_MODELS = [
    "CamemBERTThematicClassifier",
    "LogisticRegressionThematicClassifier",
    "SentenceBERTThematicClassifier", 
    "NaiveBayesThematicClassifier"
]

# Performance targets
PERFORMANCE_TARGETS = {
    "tension_detection": {
        "current_accuracy": 0.33,
        "target_accuracy": (0.75, 0.90),
        "primary_metric": "accuracy",
        "secondary_metrics": ["f1_score", "confidence_calibration"]
    },
    "thematic_classification": {
        "current_accuracy": 1.0,  # But limited scope
        "current_limitation": "Only detects 'Performance', misses 'Légitimité'",
        "target_accuracy": (0.85, 0.95),
        "target_coverage": ["Performance", "Légitimité"],
        "primary_metric": "accuracy",
        "secondary_metrics": ["f1_score", "precision", "recall"]
    }
}

# Import model classes when available
try:
    from .tension_detection import *
    from .thematic_classification import *
except ImportError:
    # Models not yet implemented
    pass
