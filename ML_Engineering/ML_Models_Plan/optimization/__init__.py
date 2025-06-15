"""
Optuna Optimization Package

This package contains Optuna-based hyperparameter optimization scripts
for all machine learning models in the hybrid classification pipeline.

Optimization Strategy:
- Test 3-4 models per classification task
- Use Optuna TPE (Tree-structured Parzen Estimator) for efficient search
- Cross-validation for robust performance estimation
- Select best performing model for production integration

Components:
- tension_optimization.py: Hyperparameter optimization for tension detection models
- thematic_optimization.py: Hyperparameter optimization for thematic classification models
"""

__version__ = "1.0.0"

import optuna
from typing import Dict, Any, List, Tuple

# Optuna configuration
OPTUNA_CONFIG = {
    "n_trials": 100,  # Number of optimization trials per model
    "timeout": 3600,  # 1 hour timeout per model optimization
    "sampler": "TPE",  # Tree-structured Parzen Estimator
    "pruner": "MedianPruner",  # Early stopping for poor trials
    "direction": "maximize",  # Maximize accuracy/F1-score
    "cv_folds": 5  # Cross-validation folds
}

# Performance tracking
OPTIMIZATION_TARGETS = {
    "tension_detection": {
        "primary_metric": "accuracy",
        "target_range": (0.75, 0.90),
        "secondary_metrics": ["f1_score", "precision", "recall"]
    },
    "thematic_classification": {
        "primary_metric": "accuracy", 
        "target_range": (0.85, 0.95),
        "secondary_metrics": ["f1_score", "precision", "recall"]
    }
}

def create_study(study_name: str, direction: str = "maximize") -> optuna.Study:
    """Create an Optuna study for hyperparameter optimization."""
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

def log_optimization_results(study: optuna.Study, model_name: str) -> Dict[str, Any]:
    """Log optimization results for analysis."""
    return {
        "model_name": model_name,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "optimization_history": [trial.value for trial in study.trials if trial.value is not None]
    }
