"""
Logistic Regression Model for Thematic Classification
====================================================

Implementation of Logistic Regression with TF-IDF features for thematic classification with Optuna optimization.

Author: ML Engineering Team
Date: 2025-06-12
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import optuna
from typing import Dict, Any, List
import logging
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThematicLogisticRegressionModel:
    """
    Logistic Regression model for thematic classification with TF-IDF features and Optuna optimization.
    
    Features:
    - TF-IDF text vectorization
    - L1, L2, and ElasticNet regularization
    - Automated hyperparameter tuning
    - Class imbalance handling
    - Feature importance analysis
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the Logistic Regression model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.pipeline = None
        self.best_params = None
        self.feature_importance = None
        self.is_trained = False
        
        logger.info("Initialized ThematicLogisticRegressionModel")
    
    def objective(self, trial: optuna.Trial, X_train: List[str], y_train: List[int],
                  X_val: List[str], y_val: List[int], class_weights: Dict[int, float]) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            class_weights: Class weights for imbalanced data
            
        Returns:
            Validation accuracy score
        """
        # Suggest hyperparameters
        C = trial.suggest_float('C', 0.01, 100.0, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
        max_iter = trial.suggest_int('max_iter', 100, 2000)
        tfidf_max_features = trial.suggest_int('tfidf_max_features', 1000, 10000)
        # Store n-gram range as string to keep Optuna happy (no tuples)
        ngram_choice = trial.suggest_categorical('tfidf_ngram_range', ['1_1', '1_2', '1_3'])
        tfidf_ngram_range = tuple(map(int, ngram_choice.split('_')))
        
        # Handle solver-penalty compatibility
        if penalty == 'elasticnet' and solver != 'saga':
            solver = 'saga'
        if penalty == 'l1' and solver == 'liblinear':
            solver = 'liblinear'
        elif penalty == 'l1' and solver != 'saga':
            solver = 'saga'
        
        # Add l1_ratio for elasticnet
        l1_ratio = None
        if penalty == 'elasticnet':
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
        
        # Create pipeline
        tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            stop_words=None,  # Keep all words for French
            lowercase=True,
            strip_accents='unicode'
        )
        
        lr_params = {
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter,
            'random_state': self.random_state,
            'class_weight': 'balanced'
        }
        
        if l1_ratio is not None:
            lr_params['l1_ratio'] = l1_ratio
        
        lr = LogisticRegression(**lr_params)
        
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', lr)
        ])
        
        # Train pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
    
    def optimize_hyperparameters(self, X_train: List[str], y_train: List[int],
                                X_val: List[str], y_val: List[int],
                                class_weights: Dict[int, float],
                                n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            class_weights: Class weights for imbalanced data
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting Logistic Regression hyperparameter optimization with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, class_weights),
            n_trials=n_trials
        )
        
        # Store best parameters
        self.best_params = study.best_params
        
        logger.info(f"Logistic Regression optimization completed. Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def train(self, X_train: List[str], y_train: List[int],
              X_val: List[str] = None, y_val: List[int] = None,
              class_weights: Dict[int, float] = None,
              optimize: bool = True, n_trials: int = 100) -> Dict[str, Any]:
        """
        Train the Logistic Regression model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
            class_weights: Class weights for imbalanced data
            optimize: Whether to run hyperparameter optimization
            n_trials: Number of optimization trials
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Logistic Regression model for thematic classification...")
        
        results = {}
        
        if optimize and X_val is not None and y_val is not None:
            # Run hyperparameter optimization
            opt_results = self.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, class_weights, n_trials
            )
            results.update(opt_results)
            
            # Use best parameters
            params = self.best_params.copy()
        else:
            # Use default parameters
            params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'max_iter': 1000,
                'tfidf_max_features': 5000,
                'tfidf_ngram_range': '1_2'
            }
        
        # Extract TF-IDF parameters
        tfidf_params = {
            'max_features': params.pop('tfidf_max_features'),
            'ngram_range': tuple(map(int, params.pop('tfidf_ngram_range').split('_'))),
            'stop_words': None,
            'lowercase': True,
            'strip_accents': 'unicode'
        }
        
        # Extract Logistic Regression parameters
        lr_params = params.copy()
        lr_params.update({
            'random_state': self.random_state,
            'class_weight': 'balanced'
        })
        
        # Create pipeline
        tfidf = TfidfVectorizer(**tfidf_params)
        lr = LogisticRegression(**lr_params)
        
        self.pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', lr)
        ])
        
        # Train the pipeline
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Get feature importance (coefficients)
        self.feature_importance = self.pipeline.named_steps['classifier'].coef_[0]
        
        # Evaluate on training set
        train_pred = self.pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        results.update({
            'train_accuracy': train_accuracy,
            'feature_importance': self.feature_importance,
            'pipeline': self.pipeline
        })
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.pipeline.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            results['val_accuracy'] = val_accuracy
            
            logger.info(f"Logistic Regression training completed - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            logger.info(f"Logistic Regression training completed - Train Acc: {train_accuracy:.4f}")
        
        return results
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Make predictions on new texts.
        
        Args:
            X: Texts to predict on
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Texts to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test: List[str], y_test: List[int], 
                class_names: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            class_names: Names of classes for reporting
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating Logistic Regression model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        logger.info(f"Logistic Regression test accuracy: {accuracy:.4f}")
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top important features (words).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # Get feature names from TF-IDF vectorizer
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(self.feature_importance)
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'pipeline': self.pipeline,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Logistic Regression model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.pipeline = model_data['pipeline']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        logger.info(f"Logistic Regression model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("Logistic Regression Thematic Classification Model - Example Usage")
    
    # This would typically be called with real data from the data preparation pipeline
    print("Logistic Regression model implementation ready for integration with data preparation pipeline")
