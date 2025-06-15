"""
Model Evaluation and Validation Package

This package contains comprehensive evaluation frameworks for validating
machine learning models against performance targets and reference data.

Components:
- performance_metrics.py: Accuracy, F1-score, precision, recall calculations
- confidence_calibration.py: Confidence score reliability assessment  
- validation_framework.py: Cross-validation and hold-out testing

Evaluation Strategy:
- Cross-Validation: 5-fold stratified CV during training
- Hold-out Test: 20% of data reserved for final evaluation
- Reference Comparison: Against data engineering target format
- A/B Testing: Hybrid vs pure rule-based performance
"""

__version__ = "1.0.0"

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Evaluation configuration
EVALUATION_CONFIG = {
    "cv_folds": 5,
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,
    "confidence_bins": 10  # For calibration analysis
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "tension_detection": {
        "accuracy_min": 0.75,
        "accuracy_target": 0.90,
        "f1_min": 0.70,
        "confidence_reliability": 0.80
    },
    "thematic_classification": {
        "accuracy_min": 0.85,
        "accuracy_target": 0.95,
        "f1_min": 0.80,
        "confidence_reliability": 0.80
    }
}

def calculate_basic_metrics(y_true: List, y_pred: List) -> Dict[str, float]:
    """Calculate basic classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted')
    }

def evaluate_against_targets(metrics: Dict[str, float], task: str) -> Dict[str, bool]:
    """Evaluate metrics against performance targets."""
    thresholds = PERFORMANCE_THRESHOLDS[task]
    return {
        "meets_accuracy_min": metrics["accuracy"] >= thresholds["accuracy_min"],
        "meets_accuracy_target": metrics["accuracy"] >= thresholds["accuracy_target"],
        "meets_f1_min": metrics["f1_score"] >= thresholds["f1_min"],
        "overall_success": (
            metrics["accuracy"] >= thresholds["accuracy_min"] and 
            metrics["f1_score"] >= thresholds["f1_min"]
        )
    }

def compare_with_baseline(ml_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> Dict[str, float]:
    """Compare ML model performance with baseline (rule-based) performance."""
    return {
        "accuracy_improvement": ml_metrics["accuracy"] - baseline_metrics["accuracy"],
        "f1_improvement": ml_metrics["f1_score"] - baseline_metrics["f1_score"],
        "relative_accuracy_gain": (ml_metrics["accuracy"] - baseline_metrics["accuracy"]) / baseline_metrics["accuracy"],
        "relative_f1_gain": (ml_metrics["f1_score"] - baseline_metrics["f1_score"]) / baseline_metrics["f1_score"]
    }
