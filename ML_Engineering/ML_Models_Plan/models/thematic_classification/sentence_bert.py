"""
Sentence-BERT Model for Thematic Classification
==============================================

Implementation of Sentence-BERT embeddings with Neural Network classifier for thematic classification.

Author: ML Engineering Team
Date: 2025-06-12
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import optuna
from typing import Dict, Any, List
import logging
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThematicNeuralNetwork(nn.Module):
    """
    Neural Network classifier for sentence embeddings.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout: float = 0.3):
        """
        Initialize the neural network.
        
        Args:
            input_size: Size of input embeddings
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(ThematicNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ThematicSentenceBERTModel:
    """
    Sentence-BERT model for thematic classification with neural network classifier.
    
    Features:
    - Multilingual sentence embeddings
    - Configurable neural network architecture
    - Automated hyperparameter tuning
    - Class imbalance handling
    - GPU support
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", random_state: int = 42):
        """
        Initialize the Sentence-BERT model.

        Args:
            model_name: Pre-trained sentence transformer model
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state

        # Initialize sentence transformer with error handling
        try:
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Successfully initialized SentenceTransformer with {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            logger.info("Falling back to lighter model...")
            try:
                # Fallback to a lighter model
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                self.sentence_model = SentenceTransformer(fallback_model)
                self.model_name = fallback_model
                logger.info(f"Successfully initialized with fallback model: {fallback_model}")
            except Exception as e2:
                logger.error(f"Failed to initialize even with fallback model: {e2}")
                raise RuntimeError(f"Cannot initialize SentenceTransformer: {e2}")

        self.classifier = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        logger.info(f"Initialized ThematicSentenceBERTModel with {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to sentence embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Sentence embeddings
        """
        embeddings = self.sentence_model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray, class_weights: torch.Tensor) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training embeddings
            y_train: Training labels
            X_val: Validation embeddings
            y_val: Validation labels
            class_weights: Class weights for imbalanced data
            
        Returns:
            Validation accuracy score
        """
        # Suggest hyperparameters
        hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
        hidden_size = trial.suggest_int('hidden_size', 64, 512)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Create hidden sizes list
        hidden_sizes = [hidden_size] * hidden_layers
        
        # Create model
        model = ThematicNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            num_classes=len(np.unique(y_train)),
            dropout=dropout
        ).to(self.device)
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.LongTensor(y_train).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.LongTensor(y_val).to(self.device)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        model.train()
        for epoch in range(20):  # Fixed number of epochs for optimization
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Validation
        model.eval()
        val_predictions = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                outputs = model(batch_X)
                predictions = torch.argmax(outputs, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
        
        accuracy = accuracy_score(y_val, val_predictions)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return accuracy
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                class_weights: torch.Tensor,
                                n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training embeddings
            y_train: Training labels
            X_val: Validation embeddings
            y_val: Validation labels
            class_weights: Class weights for imbalanced data
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting Sentence-BERT hyperparameter optimization with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, class_weights),
            n_trials=n_trials
        )
        
        # Store best parameters
        self.best_params = study.best_params
        
        logger.info(f"Sentence-BERT optimization completed. Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def train(self, X_train: List[str], y_train: List[int],
              X_val: List[str] = None, y_val: List[int] = None,
              class_weights: torch.Tensor = None,
              optimize: bool = True, n_trials: int = 50) -> Dict[str, Any]:
        """
        Train the Sentence-BERT model.
        
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
        logger.info("Training Sentence-BERT model for thematic classification...")
        
        # Encode texts to embeddings
        logger.info("Encoding training texts...")
        train_embeddings = self.encode_texts(X_train)
        train_embeddings = self.scaler.fit_transform(train_embeddings)
        
        val_embeddings = None
        if X_val is not None:
            logger.info("Encoding validation texts...")
            val_embeddings = self.encode_texts(X_val)
            val_embeddings = self.scaler.transform(val_embeddings)
        
        results = {}
        
        if optimize and val_embeddings is not None and y_val is not None:
            # Run hyperparameter optimization
            opt_results = self.optimize_hyperparameters(
                train_embeddings, np.array(y_train), val_embeddings, np.array(y_val), class_weights, n_trials
            )
            results.update(opt_results)
            
            # Use best parameters
            params = self.best_params
        else:
            # Use default parameters
            params = {
                'hidden_layers': 2,
                'hidden_size': 128,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        
        # Create final model
        hidden_sizes = [params['hidden_size']] * params['hidden_layers']
        self.classifier = ThematicNeuralNetwork(
            input_size=train_embeddings.shape[1],
            hidden_sizes=hidden_sizes,
            num_classes=len(np.unique(y_train)),
            dropout=params['dropout']
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        optimizer = optim.Adam(self.classifier.parameters(), lr=params['learning_rate'])
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_embeddings).to(self.device),
            torch.LongTensor(y_train).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        
        # Training loop
        self.classifier.train()
        train_losses = []
        
        for epoch in range(50):  # More epochs for final training
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.classifier(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {train_losses[-1]:.4f}")
        
        self.is_trained = True
        
        # Evaluate on training set
        train_pred = self.predict_embeddings(train_embeddings)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        results.update({
            'train_accuracy': train_accuracy,
            'train_losses': train_losses,
            'classifier': self.classifier
        })
        
        # Evaluate on validation set if provided
        if val_embeddings is not None and y_val is not None:
            val_pred = self.predict_embeddings(val_embeddings)
            val_accuracy = accuracy_score(y_val, val_pred)
            results['val_accuracy'] = val_accuracy
            
            logger.info(f"Sentence-BERT training completed - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            logger.info(f"Sentence-BERT training completed - Train Acc: {train_accuracy:.4f}")
        
        return results
    
    def predict_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Make predictions on embeddings.
        
        Args:
            embeddings: Sentence embeddings
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.classifier.eval()
        predictions = []
        
        with torch.no_grad():
            embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
            outputs = self.classifier(embeddings_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return predictions
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on new texts.
        
        Args:
            texts: Texts to predict on
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Encode texts and scale
        embeddings = self.encode_texts(texts)
        embeddings = self.scaler.transform(embeddings)
        
        return self.predict_embeddings(embeddings)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: Texts to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Encode texts and scale
        embeddings = self.encode_texts(texts)
        embeddings = self.scaler.transform(embeddings)
        
        self.classifier.eval()
        with torch.no_grad():
            embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
            outputs = self.classifier(embeddings_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probabilities
    
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
        
        logger.info("Evaluating Sentence-BERT model...")
        
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
        
        logger.info(f"Sentence-BERT test accuracy: {accuracy:.4f}")
        
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
            'classifier_state_dict': self.classifier.state_dict(),
            'classifier_architecture': {
                'input_size': self.classifier.network[0].in_features,
                'hidden_sizes': [layer.out_features for layer in self.classifier.network if isinstance(layer, nn.Linear)][:-1],
                'num_classes': self.classifier.network[-1].out_features,
                'dropout': 0.3  # Default value, could be stored separately
            },
            'scaler': self.scaler,
            'best_params': self.best_params,
            'model_name': self.model_name,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Sentence-BERT model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        # Restore architecture
        arch = model_data['classifier_architecture']
        self.classifier = ThematicNeuralNetwork(
            input_size=arch['input_size'],
            hidden_sizes=arch['hidden_sizes'],
            num_classes=arch['num_classes'],
            dropout=arch['dropout']
        ).to(self.device)
        
        # Load state dict
        self.classifier.load_state_dict(model_data['classifier_state_dict'])
        
        # Restore other components
        self.scaler = model_data['scaler']
        self.best_params = model_data['best_params']
        self.model_name = model_data['model_name']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        logger.info(f"Sentence-BERT model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("Sentence-BERT Thematic Classification Model - Example Usage")
    
    # This would typically be called with real data from the data preparation pipeline
    print("Sentence-BERT model implementation ready for integration with data preparation pipeline")
