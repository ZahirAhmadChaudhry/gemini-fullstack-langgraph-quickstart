"""
Random Forest Model for Tension Detection
=========================================

Implementation of Random Forest classifier for tension detection with Optuna optimization.

Author: ML Engineering Team
Date: 2025-06-12
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
from typing import Dict, Any, Tuple
import logging
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensionRandomForestModel:
    """
    Random Forest model for tension detection with Optuna hyperparameter optimization.
    
    Features:
    - Automated hyperparameter tuning
    - Feature importance analysis
    - Cross-validation support
    - Model persistence
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the Random Forest model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.is_trained = False
        
        logger.info("Initialized TensionRandomForestModel")
    
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray, class_weights: Dict[int, float]) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
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
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': self.random_state,
            'class_weight': 'balanced',  # Handle class imbalance
            'n_jobs': -1  # Use all available cores
        }
        
        # Train model with suggested parameters
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                class_weights: Dict[int, float],
                                n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
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
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, class_weights),
            n_trials=n_trials
        )
        
        # Store best parameters
        self.best_params = study.best_params
        self.best_params['random_state'] = self.random_state
        self.best_params['class_weight'] = 'balanced'
        self.best_params['n_jobs'] = -1
        
        logger.info(f"Optimization completed. Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              class_weights: Dict[int, float] = None,
              optimize: bool = True, n_trials: int = 100) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weights: Class weights for imbalanced data
            optimize: Whether to run hyperparameter optimization
            n_trials: Number of optimization trials
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Random Forest model for tension detection...")
        
        results = {}
        
        if optimize and X_val is not None and y_val is not None:
            # Run hyperparameter optimization
            opt_results = self.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, class_weights, n_trials
            )
            results.update(opt_results)
            
            # Train final model with best parameters
            self.model = RandomForestClassifier(**self.best_params)
        else:
            # Use default parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        results.update({
            'train_accuracy': train_accuracy,
            'feature_importance': self.feature_importance,
            'model': self.model
        })
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            results['val_accuracy'] = val_accuracy
            
            logger.info(f"Training completed - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            logger.info(f"Training completed - Train Acc: {train_accuracy:.4f}")
        
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
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                class_names: list = None) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: Names of classes for reporting
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating Random Forest model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Prepare classification report safely. Handle cases where some classes are
        # missing in the test set (causing a mismatch between the number of
        # provided ``class_names`` and the detected labels).
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
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            feature_names: Names of features
            
        Returns:
            DataFrame with feature importance rankings
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("Random Forest Tension Detection Model - Example Usage")
    
    # This would typically be called with real data from the data preparation pipeline
    print("Model implementation ready for integration with data preparation pipeline")
