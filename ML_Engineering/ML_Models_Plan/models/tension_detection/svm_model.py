"""
SVM Model for Tension Detection
===============================

Implementation of Support Vector Machine classifier for tension detection with Optuna optimization.

Author: ML Engineering Team
Date: 2025-06-12
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
from typing import Dict, Any, Tuple
import logging
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensionSVMModel:
    """
    Support Vector Machine model for tension detection with Optuna hyperparameter optimization.
    
    Features:
    - RBF, polynomial, and sigmoid kernel support
    - Automated hyperparameter tuning
    - Feature scaling for optimal performance
    - Model persistence
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the SVM model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.is_trained = False
        
        logger.info("Initialized TensionSVMModel")
    
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray, class_weights: Dict[int, float]) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training features (scaled)
            y_train: Training labels
            X_val: Validation features (scaled)
            y_val: Validation labels
            class_weights: Class weights for imbalanced data
            
        Returns:
            Validation accuracy score
        """
        # Suggest hyperparameters
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        C = trial.suggest_float('C', 0.1, 100.0, log=True)
        gamma = trial.suggest_float('gamma', 0.001, 1.0, log=True)
        
        params = {
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'random_state': self.random_state,
            'class_weight': 'balanced'  # Handle class imbalance
        }
        
        # Add degree parameter for polynomial kernel
        if kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
        
        # Train model with suggested parameters
        model = SVC(**params)
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
        logger.info(f"Starting SVM hyperparameter optimization with {n_trials} trials...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Replace any potential NaNs/Infs that could appear after scaling (e.g.,
        # due to zero variance columns) with zeros. This guarantees downstream
        # models such as SVC receive finite values only.
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, class_weights),
            n_trials=n_trials
        )
        
        # Store best parameters
        self.best_params = study.best_params
        self.best_params['random_state'] = self.random_state
        self.best_params['class_weight'] = 'balanced'
        
        logger.info(f"SVM optimization completed. Best accuracy: {study.best_value:.4f}")
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
        Train the SVM model.
        
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
        logger.info("Training SVM model for tension detection...")
        
        results = {}
        
        if optimize and X_val is not None and y_val is not None:
            # Run hyperparameter optimization
            opt_results = self.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, class_weights, n_trials
            )
            results.update(opt_results)
            
            # Train final model with best parameters
            self.model = SVC(**self.best_params, probability=True)  # Enable probability estimates
        else:
            # Use default parameters
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                random_state=self.random_state,
                probability=True
            )
        
        # Scale features and train the model (ensure finite values)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        results.update({
            'train_accuracy': train_accuracy,
            'model': self.model,
            'scaler': self.scaler
        })
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            val_pred = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, val_pred)
            results['val_accuracy'] = val_accuracy
            
            logger.info(f"SVM training completed - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            logger.info(f"SVM training completed - Train Acc: {train_accuracy:.4f}")
        
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
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
        
        logger.info("Evaluating SVM model...")
        
        # Scale features and make predictions
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Robust classification report generation (handles missing classes in test).
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
        
        logger.info(f"SVM test accuracy: {accuracy:.4f}")
        
        return results
    
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
            'scaler': self.scaler,
            'best_params': self.best_params,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"SVM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_params = model_data['best_params']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        logger.info(f"SVM model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("SVM Tension Detection Model - Example Usage")
    
    # This would typically be called with real data from the data preparation pipeline
    print("SVM model implementation ready for integration with data preparation pipeline")
