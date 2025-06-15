"""
Ensemble Model for Tension Detection
====================================

Implementation of Voting Classifier ensemble combining Random Forest, XGBoost, and SVM
for robust tension detection with Optuna optimization.

Author: ML Engineering Team
Date: 2025-06-12
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
from typing import Dict, Any, Tuple
import logging
import joblib
from pathlib import Path

# Import base models
from .random_forest import TensionRandomForestModel
from .xgboost_model import TensionXGBoostModel
from .svm_model import TensionSVMModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensionEnsembleModel:
    """
    Ensemble model combining Random Forest, XGBoost, and SVM for tension detection.
    
    Features:
    - Soft and hard voting strategies
    - Weighted voting with Optuna optimization
    - Individual model confidence tracking
    - Robust predictions through model diversity
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ensemble model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.rf_model = TensionRandomForestModel(random_state)
        self.xgb_model = TensionXGBoostModel(random_state)
        self.svm_model = TensionSVMModel(random_state)
        self.ensemble = None
        self.best_params = None
        self.is_trained = False
        
        logger.info("Initialized TensionEnsembleModel")
    
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray, class_weights: Dict[int, float]) -> float:
        """
        Optuna objective function for ensemble optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weights: Class weights for imbalanced data
            
        Returns:
            Validation accuracy score
        """
        # Suggest ensemble parameters
        voting = trial.suggest_categorical('voting', ['soft', 'hard'])
        rf_weight = trial.suggest_float('rf_weight', 0.1, 1.0)
        xgb_weight = trial.suggest_float('xgb_weight', 0.1, 1.0)
        svm_weight = trial.suggest_float('svm_weight', 0.1, 1.0)
        
        # Normalize weights
        total_weight = rf_weight + xgb_weight + svm_weight
        rf_weight /= total_weight
        xgb_weight /= total_weight
        svm_weight /= total_weight
        
        # Create ensemble with suggested parameters
        estimators = [
            ('rf', self.rf_model.model),
            ('xgb', self.xgb_model.model),
            ('svm', self.svm_model.model)
        ]
        
        weights = [rf_weight, xgb_weight, svm_weight]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred = ensemble.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
    
    def optimize_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         class_weights: Dict[int, float],
                         n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize ensemble parameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weights: Class weights for imbalanced data
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting ensemble optimization with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, class_weights),
            n_trials=n_trials
        )
        
        # Store best parameters
        self.best_params = study.best_params
        
        logger.info(f"Ensemble optimization completed. Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              class_weights: Dict[int, float] = None,
              optimize: bool = True, n_trials: int = 50) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weights: Class weights for imbalanced data
            optimize: Whether to run ensemble optimization
            n_trials: Number of optimization trials
            
        Returns:
            Training results dictionary
        """
        logger.info("Training ensemble model for tension detection...")
        
        results = {}
        
        # First, train individual models with reduced trials for efficiency
        logger.info("Training individual models...")
        
        # Train Random Forest
        rf_results = self.rf_model.train(
            X_train, y_train, X_val, y_val, class_weights, 
            optimize=True, n_trials=20
        )
        
        # Train XGBoost
        xgb_results = self.xgb_model.train(
            X_train, y_train, X_val, y_val, class_weights,
            optimize=True, n_trials=20
        )
        
        # Train SVM
        svm_results = self.svm_model.train(
            X_train, y_train, X_val, y_val, class_weights,
            optimize=True, n_trials=20
        )
        
        results['individual_models'] = {
            'random_forest': rf_results,
            'xgboost': xgb_results,
            'svm': svm_results
        }
        
        # Create ensemble
        estimators = [
            ('rf', self.rf_model.model),
            ('xgb', self.xgb_model.model),
            ('svm', self.svm_model.model)
        ]
        
        if optimize and X_val is not None and y_val is not None:
            # Optimize ensemble parameters
            opt_results = self.optimize_ensemble(
                X_train, y_train, X_val, y_val, class_weights, n_trials
            )
            results.update(opt_results)
            
            # Create optimized ensemble
            weights = [
                self.best_params['rf_weight'],
                self.best_params['xgb_weight'],
                self.best_params['svm_weight']
            ]
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting=self.best_params['voting'],
                weights=weights
            )
        else:
            # Use default ensemble parameters
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=[0.4, 0.4, 0.2]  # Slightly favor RF and XGB
            )
        
        # Train ensemble
        self.ensemble.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_pred = self.ensemble.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        results.update({
            'train_accuracy': train_accuracy,
            'ensemble': self.ensemble
        })
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.ensemble.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            results['val_accuracy'] = val_accuracy
            
            logger.info(f"Ensemble training completed - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            logger.info(f"Ensemble training completed - Train Acc: {train_accuracy:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        # Native probability support only exists when voting='soft'.
        if hasattr(self.ensemble, "predict_proba"):
            try:
                return self.ensemble.predict_proba(X)
            except AttributeError:
                pass  # fall through to manual aggregation

        # Manual aggregation: average probabilities of base models that expose predict_proba
        probas = []
        for model in [self.rf_model, self.xgb_model, self.svm_model]:
            if model is not None and hasattr(model, "predict_proba"):
                probas.append(model.predict_proba(X))

        if probas:
            # Ensure same shape, then average
            return np.mean(probas, axis=0)
        else:
            # As a last resort, convert hard predictions to one-hot probabilities
            preds = self.predict(X)
            n_classes = len(np.unique(preds))
            proba = np.zeros((len(preds), n_classes))
            proba[np.arange(len(preds)), preds] = 1.0
            return proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                class_names: list = None) -> Dict[str, Any]:
        """
        Evaluate the ensemble model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: Names of classes for reporting
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        logger.info("Evaluating ensemble model...")
        
        # Make predictions
        y_pred = self.predict(X_test)

        # Probabilities may be unavailable for hard voting; handle gracefully
        try:
            y_proba = self.predict_proba(X_test)
        except Exception:
            y_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        if class_names is not None:
            try:
                class_report = classification_report(
                    y_test,
                    y_pred,
                    labels=list(range(len(class_names))),
                    target_names=[str(c) for c in class_names],
                    output_dict=True,
                    zero_division=0,
                )
            except ValueError as e:
                logger.warning(
                    "classification_report raised ValueError (%s). Falling back to auto-generated class names.",
                    e,
                )
                class_report = classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                )
        else:
            class_report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            )
        
        # Evaluate individual models for comparison
        individual_results = {}
        individual_results['rf'] = self.rf_model.evaluate(X_test, y_test, class_names)
        individual_results['xgb'] = self.xgb_model.evaluate(X_test, y_test, class_names)
        individual_results['svm'] = self.svm_model.evaluate(X_test, y_test, class_names)
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_proba,
            'individual_models': individual_results
        }
        
        logger.info(f"Ensemble test accuracy: {accuracy:.4f}")
        
        return results
    
    def save_model(self, filepath: str):
        """
        Save the trained ensemble model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving")
        
        model_data = {
            'ensemble': self.ensemble,
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'svm_model': self.svm_model,
            'best_params': self.best_params,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained ensemble model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.ensemble = model_data['ensemble']
        self.rf_model = model_data['rf_model']
        self.xgb_model = model_data['xgb_model']
        self.svm_model = model_data['svm_model']
        self.best_params = model_data['best_params']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        logger.info(f"Ensemble model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("Ensemble Tension Detection Model - Example Usage")
    
    # This would typically be called with real data from the data preparation pipeline
    print("Ensemble model implementation ready for integration with data preparation pipeline")
