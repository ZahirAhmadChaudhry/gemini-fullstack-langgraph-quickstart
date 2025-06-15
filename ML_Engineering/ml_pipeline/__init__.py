"""
ML Pipeline package for advanced machine learning capabilities.

This package provides unsupervised learning, evaluation, and dataset management
capabilities for the French Sustainability Opinion Analysis system.
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"

# Import main components for easy access (lazy imports to avoid numba compilation delays)
def _lazy_import():
    """Lazy import to avoid slow numba compilation on module import."""
    try:
        from .unsupervised_learning import TopicModeling, SemanticSearch, FeatureEngineering
        from .evaluation import MetricsCalculator
        from .dataset_management import DataSplitter
        return {
            "TopicModeling": TopicModeling,
            "SemanticSearch": SemanticSearch,
            "FeatureEngineering": FeatureEngineering,
            "MetricsCalculator": MetricsCalculator,
            "DataSplitter": DataSplitter
        }
    except ImportError as e:
        print(f"Warning: Some ML components not available: {e}")
        return {}

# Make components available at module level
_components = _lazy_import()
globals().update(_components)

__all__ = list(_components.keys())
