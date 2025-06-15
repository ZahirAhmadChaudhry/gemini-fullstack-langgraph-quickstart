"""
Tension Detection Models Package

This package contains machine learning models for improving tension detection
from the current 33% accuracy to the target 75-90% accuracy range.

Models included:
1. RandomForestTensionDetector - Interpretable ensemble method
2. XGBoostTensionDetector - Gradient boosting for complex patterns  
3. SVMTensionDetector - Non-linear boundary detection
4. EnsembleTensionDetector - Combined approach for robustness

Current Problem:
- Simple max strength selection achieves only 33% accuracy
- Need to leverage rich features from enhanced data engineering pipeline

Target Performance:
- Accuracy: 75-90% (vs current 33%)
- F1-Score: 0.75+ across all tension types
- Confidence Calibration: 80%+ reliability
"""

__version__ = "1.0.0"

# Import implemented models
from .random_forest import TensionRandomForestModel
from .xgboost_model import TensionXGBoostModel

# Import models to be implemented
try:
    from .svm_model import TensionSVMModel
except ImportError:
    TensionSVMModel = None

try:
    from .ensemble_model import TensionEnsembleModel
except ImportError:
    TensionEnsembleModel = None

__all__ = [
    'TensionRandomForestModel',
    'TensionXGBoostModel',
    'TensionSVMModel',
    'TensionEnsembleModel'
]

TENSION_TYPES = [
    "accumulation_partage",
    "croissance_decroissance", 
    "individuel_collectif",
    "local_global",
    "court_terme_long_terme"
]

FEATURE_GROUPS = {
    "tension_patterns": "Pre-computed strength scores from data engineering",
    "discourse_markers": "Linguistic indicators of tension",
    "pos_distribution": "Part-of-speech patterns",
    "noun_phrases": "Concept indicators", 
    "sustainability_scores": "Domain relevance scores",
    "temporal_confidence": "Time-based context",
    "lexical_diversity": "Text complexity measures"
}
