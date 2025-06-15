"""
Naive Bayes Model for Thematic Classification
============================================

Implementation of Multinomial Naive Bayes with feature engineering for thematic classification.

Author: ML Engineering Team
Date: 2025-06-12
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
from typing import Dict, Any, List
import logging
import joblib
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThematicNaiveBayesModel:
    """
    Naive Bayes model for thematic classification with feature engineering and Optuna optimization.
    
    Features:
    - TF-IDF and Count vectorization
    - Feature selection with chi-squared test
    - Domain-specific feature engineering
    - Automated hyperparameter tuning
    - Class imbalance handling
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the Naive Bayes model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.pipeline = None
        self.best_params = None
        self.is_trained = False
        
        # Domain-specific keywords for feature engineering
        self.performance_keywords = [
            'performance', 'efficacité', 'productivité', 'rendement', 'résultat',
            'objectif', 'cible', 'mesure', 'indicateur', 'amélioration'
        ]
        
        self.legitimacy_keywords = [
            'légitimité', 'acceptabilité', 'confiance', 'crédibilité', 'autorité',
            'reconnaissance', 'validation', 'approbation', 'consensus', 'adhésion'
        ]
        
        logger.info("Initialized ThematicNaiveBayesModel")
    
    def extract_domain_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract domain-specific features from texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Domain feature matrix
        """
        features = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Count performance-related keywords
            perf_count = sum(1 for keyword in self.performance_keywords if keyword in text_lower)
            
            # Count legitimacy-related keywords
            legit_count = sum(1 for keyword in self.legitimacy_keywords if keyword in text_lower)
            
            # Text length features
            text_length = len(text)
            word_count = len(text.split())
            
            # Punctuation features
            question_marks = text.count('?')
            exclamation_marks = text.count('!')
            
            # Sustainability-related patterns
            sustainability_patterns = ['durable', 'environnement', 'écolog', 'vert', 'renouvelable']
            sustainability_count = sum(1 for pattern in sustainability_patterns if pattern in text_lower)
            
            features.append([
                perf_count, legit_count, text_length, word_count,
                question_marks, exclamation_marks, sustainability_count
            ])
        
        return np.array(features)
    
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
        alpha = trial.suggest_float('alpha', 0.1, 10.0)
        fit_prior = trial.suggest_categorical('fit_prior', [True, False])
        vectorizer_type = trial.suggest_categorical('vectorizer_type', ['tfidf', 'count'])
        max_features = trial.suggest_int('max_features', 1000, 10000)
        feature_selection_k_raw = trial.suggest_int('feature_selection_k', 100, 5000)
        feature_selection_k = min(feature_selection_k_raw, max_features)
        # Encode ngram_range as string to silence Optuna tuple warning
        ngram_choice = trial.suggest_categorical('ngram_range', ['1_1', '1_2', '1_3'])
        ngram_range = tuple(map(int, ngram_choice.split('_')))
        
        # Create vectorizer
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words=None,
                lowercase=True,
                strip_accents='unicode'
            )
        else:
            vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words=None,
                lowercase=True,
                strip_accents='unicode'
            )
        
        # Create feature extraction pipeline
        text_features = Pipeline([
            ('vectorizer', vectorizer),
            ('feature_selection', SelectKBest(chi2, k=feature_selection_k))
        ])
        
        # Domain features transformer
        domain_transformer = FunctionTransformer(
            self.extract_domain_features,
            validate=False
        )
        
        # Combine features
        feature_union = FeatureUnion([
            ('text_features', text_features),
            ('domain_features', domain_transformer)
        ])
        
        # Create full pipeline
        pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', MultinomialNB(alpha=alpha, fit_prior=fit_prior))
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
        logger.info(f"Starting Naive Bayes hyperparameter optimization with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, class_weights),
            n_trials=n_trials
        )
        
        # Store best parameters
        self.best_params = study.best_params
        
        logger.info(f"Naive Bayes optimization completed. Best accuracy: {study.best_value:.4f}")
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
        Train the Naive Bayes model.
        
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
        logger.info("Training Naive Bayes model for thematic classification...")
        
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
                'alpha': 1.0,
                'fit_prior': True,
                'feature_selection_k': 2000,
                'vectorizer_type': 'tfidf',
                'max_features': 5000,
                'ngram_range': (1, 2)
            }
        
        # Create vectorizer
        if params['vectorizer_type'] == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=params['max_features'],
                ngram_range=params['ngram_range'],
                stop_words=None,
                lowercase=True,
                strip_accents='unicode'
            )
        else:
            vectorizer = CountVectorizer(
                max_features=params['max_features'],
                ngram_range=params['ngram_range'],
                stop_words=None,
                lowercase=True,
                strip_accents='unicode'
            )
        
        # Create feature extraction pipeline
        text_features = Pipeline([
            ('vectorizer', vectorizer),
            ('feature_selection', SelectKBest(chi2, k=params['feature_selection_k']))
        ])
        
        # Domain features transformer
        domain_transformer = FunctionTransformer(
            self.extract_domain_features,
            validate=False
        )
        
        # Combine features
        feature_union = FeatureUnion([
            ('text_features', text_features),
            ('domain_features', domain_transformer)
        ])
        
        # Create full pipeline
        self.pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', MultinomialNB(
                alpha=params['alpha'],
                fit_prior=params['fit_prior']
            ))
        ])
        
        # Train the pipeline
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_pred = self.pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        results.update({
            'train_accuracy': train_accuracy,
            'pipeline': self.pipeline
        })
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.pipeline.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            results['val_accuracy'] = val_accuracy
            
            logger.info(f"Naive Bayes training completed - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            logger.info(f"Naive Bayes training completed - Train Acc: {train_accuracy:.4f}")
        
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
        
        logger.info("Evaluating Naive Bayes model...")
        
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
        
        logger.info(f"Naive Bayes test accuracy: {accuracy:.4f}")
        
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
            'pipeline': self.pipeline,
            'best_params': self.best_params,
            'performance_keywords': self.performance_keywords,
            'legitimacy_keywords': self.legitimacy_keywords,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Naive Bayes model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.pipeline = model_data['pipeline']
        self.best_params = model_data['best_params']
        self.performance_keywords = model_data['performance_keywords']
        self.legitimacy_keywords = model_data['legitimacy_keywords']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        logger.info(f"Naive Bayes model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("Naive Bayes Thematic Classification Model - Example Usage")
    
    # This would typically be called with real data from the data preparation pipeline
    print("Naive Bayes model implementation ready for integration with data preparation pipeline")
