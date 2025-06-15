"""
Cross-validation module for ML model evaluation.

This module provides cross-validation capabilities for evaluating
ML models and hyperparameter optimization.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import random

try:
    from sklearn.model_selection import KFold, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KFold = None
    StratifiedKFold = None

# Configure logging
logger = logging.getLogger(__name__)

class CrossValidator:
    """
    Cross-validation system for ML model evaluation.
    """
    
    def __init__(self, n_splits: int = 5, random_seed: int = 42):
        """
        Initialize the cross-validator.
        
        Args:
            n_splits: Number of cross-validation folds
            random_seed: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Using simple cross-validation.")
    
    def simple_k_fold_split(self, data: List[Any], 
                           stratify_labels: Optional[List[str]] = None) -> List[Tuple[List[int], List[int]]]:
        """
        Simple k-fold split implementation when sklearn is not available.
        
        Args:
            data: List of data items
            stratify_labels: Optional labels for stratification
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_samples = len(data)
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        folds = []
        
        for i in range(self.n_splits):
            start_idx = i * fold_size
            if i == self.n_splits - 1:  # Last fold gets remaining samples
                end_idx = n_samples
            else:
                end_idx = (i + 1) * fold_size
            
            val_indices = indices[start_idx:end_idx]
            train_indices = indices[:start_idx] + indices[end_idx:]
            
            folds.append((train_indices, val_indices))
        
        return folds
    
    def cross_validate_segments(self, segments: List[Dict[str, Any]], 
                               evaluation_func: Callable,
                               stratify_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform cross-validation on segments.
        
        Args:
            segments: List of segments to validate
            evaluation_func: Function to evaluate each fold
            stratify_by: Optional field to stratify by
            
        Returns:
            Cross-validation results
        """
        if not segments:
            return {"error": "No segments provided"}
        
        logger.info(f"Starting {self.n_splits}-fold cross-validation on {len(segments)} segments")
        
        # Prepare stratification labels if needed
        stratify_labels = None
        if stratify_by:
            stratify_labels = []
            for segment in segments:
                if stratify_by == "temporal_context":
                    label = segment.get("features", {}).get("temporal_context", "unknown")
                elif stratify_by == "source_doc_id":
                    label = segment.get("source_doc_id", "unknown")
                else:
                    label = segment.get(stratify_by, "unknown")
                stratify_labels.append(label)
        
        # Get cross-validation splits
        if SKLEARN_AVAILABLE and stratify_labels:
            # Use sklearn stratified k-fold
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            cv_splits = list(skf.split(segments, stratify_labels))
        elif SKLEARN_AVAILABLE:
            # Use sklearn regular k-fold
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            cv_splits = list(kf.split(segments))
        else:
            # Use simple implementation
            cv_splits = self.simple_k_fold_split(segments, stratify_labels)
        
        # Perform cross-validation
        fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(cv_splits):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_splits}")
            
            # Split data
            train_segments = [segments[i] for i in train_indices]
            val_segments = [segments[i] for i in val_indices]
            
            # Evaluate fold
            try:
                fold_result = evaluation_func(train_segments, val_segments, fold_idx)
                fold_result["fold"] = fold_idx
                fold_result["train_size"] = len(train_segments)
                fold_result["val_size"] = len(val_segments)
                fold_results.append(fold_result)
                
                logger.info(f"Fold {fold_idx + 1} completed successfully")
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx + 1}: {e}")
                fold_results.append({
                    "fold": fold_idx,
                    "error": str(e),
                    "train_size": len(train_segments),
                    "val_size": len(val_segments)
                })
        
        # Aggregate results
        cv_results = self._aggregate_fold_results(fold_results)
        cv_results["n_splits"] = self.n_splits
        cv_results["total_segments"] = len(segments)
        cv_results["stratify_by"] = stratify_by
        
        logger.info("Cross-validation completed")
        return cv_results
    
    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from all folds.
        
        Args:
            fold_results: List of results from each fold
            
        Returns:
            Aggregated cross-validation results
        """
        if not fold_results:
            return {"error": "No fold results to aggregate"}
        
        # Filter out error folds
        valid_folds = [fold for fold in fold_results if "error" not in fold]
        error_folds = [fold for fold in fold_results if "error" in fold]
        
        if not valid_folds:
            return {
                "error": "All folds failed",
                "failed_folds": len(error_folds),
                "fold_errors": error_folds
            }
        
        # Collect metrics from valid folds
        metrics_by_fold = {}
        for fold in valid_folds:
            for key, value in fold.items():
                if key not in ["fold", "train_size", "val_size"] and isinstance(value, (int, float)):
                    if key not in metrics_by_fold:
                        metrics_by_fold[key] = []
                    metrics_by_fold[key].append(value)
        
        # Calculate statistics for each metric
        aggregated_metrics = {}
        for metric_name, values in metrics_by_fold.items():
            if values:
                aggregated_metrics[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "values": values
                }
        
        # Overall statistics
        train_sizes = [fold.get("train_size", 0) for fold in valid_folds]
        val_sizes = [fold.get("val_size", 0) for fold in valid_folds]
        
        results = {
            "aggregated_metrics": aggregated_metrics,
            "fold_statistics": {
                "successful_folds": len(valid_folds),
                "failed_folds": len(error_folds),
                "avg_train_size": float(np.mean(train_sizes)) if train_sizes else 0,
                "avg_val_size": float(np.mean(val_sizes)) if val_sizes else 0
            },
            "individual_fold_results": fold_results
        }
        
        if error_folds:
            results["fold_errors"] = error_folds
        
        return results
    
    def hyperparameter_search(self, segments: List[Dict[str, Any]],
                             param_grid: Dict[str, List[Any]],
                             evaluation_func: Callable,
                             scoring_metric: str = "accuracy") -> Dict[str, Any]:
        """
        Perform hyperparameter search using cross-validation.
        
        Args:
            segments: List of segments for evaluation
            param_grid: Dictionary of parameter names and values to try
            evaluation_func: Function to evaluate each parameter combination
            scoring_metric: Metric to optimize
            
        Returns:
            Hyperparameter search results
        """
        logger.info("Starting hyperparameter search with cross-validation")
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        search_results = []
        
        for param_idx, params in enumerate(param_combinations):
            logger.info(f"Testing parameter combination {param_idx + 1}/{len(param_combinations)}: {params}")
            
            # Create evaluation function with current parameters
            def param_evaluation_func(train_segs, val_segs, fold_idx):
                return evaluation_func(train_segs, val_segs, fold_idx, **params)
            
            # Perform cross-validation for this parameter combination
            cv_result = self.cross_validate_segments(segments, param_evaluation_func)
            
            # Extract scoring metric
            score = None
            if "aggregated_metrics" in cv_result and scoring_metric in cv_result["aggregated_metrics"]:
                score = cv_result["aggregated_metrics"][scoring_metric]["mean"]
            
            search_results.append({
                "parameters": params,
                "cv_score": score,
                "cv_results": cv_result
            })
        
        # Find best parameters
        valid_results = [r for r in search_results if r["cv_score"] is not None]
        
        if valid_results:
            best_result = max(valid_results, key=lambda x: x["cv_score"])
            best_params = best_result["parameters"]
            best_score = best_result["cv_score"]
        else:
            best_params = None
            best_score = None
        
        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "all_results": search_results,
            "n_combinations_tested": len(param_combinations),
            "scoring_metric": scoring_metric
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters from grid."""
        if not param_grid:
            return [{}]
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = []
        
        def generate_recursive(current_params, param_idx):
            if param_idx >= len(param_names):
                combinations.append(current_params.copy())
                return
            
            param_name = param_names[param_idx]
            for value in param_values[param_idx]:
                current_params[param_name] = value
                generate_recursive(current_params, param_idx + 1)
                del current_params[param_name]
        
        generate_recursive({}, 0)
        return combinations
