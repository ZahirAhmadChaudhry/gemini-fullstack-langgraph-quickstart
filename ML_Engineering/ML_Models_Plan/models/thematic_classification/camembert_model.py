"""
CamemBERT Model for Thematic Classification
==========================================

Implementation of CamemBERT fine-tuned model for French thematic classification with Optuna optimization.

Author: ML Engineering Team
Date: 2025-06-12
"""

import numpy as np
import pandas as pd
import torch
try:
    from transformers import (
        CamembertTokenizer, CamembertForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    logger.warning("Accelerate not available - CamemBERT will use CPU only")
    ACCELERATE_AVAILABLE = False
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
from typing import Dict, Any, List
import logging
from pathlib import Path
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThematicDataset(Dataset):
    """
    Custom dataset for CamemBERT thematic classification.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text segments
            labels: List of corresponding labels
            tokenizer: CamemBERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ThematicCamemBERTModel:
    """
    CamemBERT model for thematic classification with Optuna hyperparameter optimization.
    
    Features:
    - French language understanding with CamemBERT
    - Fine-tuning for thematic classification
    - Automated hyperparameter tuning
    - Class imbalance handling with weighted loss
    - Early stopping and model checkpointing
    """
    
    def __init__(self, num_classes: int = 2, model_name: str = "camembert-base"):
        """
        Initialize the CamemBERT model.

        Args:
            num_classes: Number of classes (Performance/Légitimité)
            model_name: Pre-trained CamemBERT model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Please install with: pip install transformers>=4.41.0")

        self.num_classes = num_classes
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.best_params = None
        self.is_trained = False

        # Initialize tokenizer with error handling
        try:
            self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
            logger.info(f"Successfully initialized CamemBERT tokenizer")
        except Exception as e:
            logger.error(f"Failed to initialize CamemBERT tokenizer: {e}")
            raise RuntimeError(f"Cannot initialize CamemBERT tokenizer: {e}")

        logger.info(f"Initialized ThematicCamemBERTModel with {num_classes} classes")
        if not ACCELERATE_AVAILABLE:
            logger.warning("Accelerate not available - training will be slower and may fail on large models")
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation.
        
        Args:
            eval_pred: Predictions and labels from trainer
            
        Returns:
            Dictionary with computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {'accuracy': accuracy}
    
    def objective(self, trial: optuna.Trial, train_dataset: ThematicDataset,
                  val_dataset: ThematicDataset, class_weights: torch.Tensor) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            train_dataset: Training dataset
            val_dataset: Validation dataset
            class_weights: Class weights for imbalanced data
            
        Returns:
            Validation accuracy score
        """
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        num_epochs = trial.suggest_int('num_epochs', 3, 10)
        warmup_steps = trial.suggest_int('warmup_steps', 100, 1000)
        weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1)
        
        # Initialize model
        model = CamembertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        # Set class weights if provided
        if class_weights is not None:
            model.config.class_weights = class_weights
        
        # Build arguments dict for TrainingArguments
        _ta_kwargs = dict(
            output_dir=f'./tmp/camembert_trial_{trial.number}',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=f'./tmp/logs_trial_{trial.number}',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,
            remove_unused_columns=False,
            push_to_hub=False,
        )
        
        # Filter kwargs against the actual TrainingArguments signature to keep
        # only supported parameters.  This removes the need for a retry loop
        # and avoids silent mismatches (e.g., evaluation_strategy).
        from transformers import TrainingArguments
        import inspect as _inspect
        _allowed = set(_inspect.signature(TrainingArguments).parameters)
        _filtered_kwargs = {k: v for k, v in _ta_kwargs.items() if k in _allowed}

        # Safety: if evaluation_strategy was trimmed but load_best_model_at_end
        # survived, disable load_best_model_at_end to prevent the mismatch error.
        if 'evaluation_strategy' not in _filtered_kwargs and 'load_best_model_at_end' in _filtered_kwargs:
            _filtered_kwargs['load_best_model_at_end'] = False
            _filtered_kwargs.pop('metric_for_best_model', None)
            _filtered_kwargs.pop('greater_is_better', None)

        training_args = TrainingArguments(**_filtered_kwargs)
        
        # Initialize trainer with safer callback handling
        callbacks = None
        try:
            if (hasattr(training_args, "metric_for_best_model") and
                training_args.metric_for_best_model and
                hasattr(training_args, "evaluation_strategy") and
                training_args.evaluation_strategy != "no"):
                callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
                logger.info("Added EarlyStoppingCallback")
        except Exception as e:
            logger.warning(f"Could not add EarlyStoppingCallback: {e}. Training will continue without early stopping.")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        # Train model
        trainer.train()
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        accuracy = eval_results['eval_accuracy']
        
        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return accuracy
    
    def optimize_hyperparameters(self, train_texts: List[str], train_labels: List[int],
                                val_texts: List[str], val_labels: List[int],
                                class_weights: torch.Tensor = None,
                                n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            class_weights: Class weights for imbalanced data
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting CamemBERT hyperparameter optimization with {n_trials} trials...")
        
        # Create datasets
        train_dataset = ThematicDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = ThematicDataset(val_texts, val_labels, self.tokenizer)
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, train_dataset, val_dataset, class_weights),
            n_trials=n_trials
        )
        
        # Store best parameters
        self.best_params = study.best_params
        
        logger.info(f"CamemBERT optimization completed. Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str] = None, val_labels: List[int] = None,
              class_weights: torch.Tensor = None,
              optimize: bool = True, n_trials: int = 20) -> Dict[str, Any]:
        """
        Train the CamemBERT model.

        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            class_weights: Class weights for imbalanced data
            optimize: Whether to run hyperparameter optimization
            n_trials: Number of optimization trials

        Returns:
            Training results dictionary
        """
        logger.info("Training CamemBERT model for thematic classification...")

        # Check memory and system constraints for Windows
        import psutil
        import platform

        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        is_windows = platform.system() == "Windows"

        if is_windows and available_memory_gb < 8:
            logger.warning(f"Low memory detected ({available_memory_gb:.1f}GB). CamemBERT may fail on Windows.")
            logger.warning("Consider using lighter models (MiniLM-SVM, Logistic Regression, or Naive Bayes)")

        if not ACCELERATE_AVAILABLE and is_windows:
            logger.warning("Accelerate not available on Windows - CamemBERT training may be unstable")

        # Reduce trials for Windows to avoid memory issues
        if is_windows and n_trials > 10:
            original_trials = n_trials
            n_trials = min(10, n_trials)
            logger.info(f"Reduced optimization trials from {original_trials} to {n_trials} for Windows compatibility")

        results = {}
        
        if optimize and val_texts is not None and val_labels is not None:
            # Run hyperparameter optimization
            opt_results = self.optimize_hyperparameters(
                train_texts, train_labels, val_texts, val_labels, class_weights, n_trials
            )
            results.update(opt_results)
            
            # Use best parameters
            training_params = self.best_params
        else:
            # Use default parameters
            training_params = {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'num_epochs': 5,
                'warmup_steps': 500,
                'weight_decay': 0.01
            }
        
        # Initialize model
        self.model = CamembertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        # Set class weights if provided
        if class_weights is not None:
            self.model.config.class_weights = class_weights
        
        # Create datasets
        train_dataset = ThematicDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = None
        if val_texts is not None and val_labels is not None:
            val_dataset = ThematicDataset(val_texts, val_labels, self.tokenizer)
        
        # Training arguments (build dict first for compatibility filtering)
        _ta_kwargs = dict(
            output_dir='./camembert_thematic',
            num_train_epochs=training_params['num_epochs'],
            per_device_train_batch_size=training_params['batch_size'],
            per_device_eval_batch_size=training_params['batch_size'],
            warmup_steps=training_params['warmup_steps'],
            weight_decay=training_params['weight_decay'],
            learning_rate=training_params['learning_rate'],
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="accuracy" if val_dataset else None,
            greater_is_better=True,
            save_total_limit=1,
            remove_unused_columns=False,
            push_to_hub=False,
        )

        # Filter kwargs against the actual TrainingArguments signature to keep
        # only supported parameters.  This removes the need for a retry loop
        # and avoids silent mismatches (e.g., evaluation_strategy).
        from transformers import TrainingArguments
        import inspect as _inspect
        _allowed = set(_inspect.signature(TrainingArguments).parameters)
        _filtered_kwargs = {k: v for k, v in _ta_kwargs.items() if k in _allowed}

        # Safety: if evaluation_strategy was trimmed but load_best_model_at_end
        # survived, disable load_best_model_at_end to prevent the mismatch error.
        if 'evaluation_strategy' not in _filtered_kwargs and 'load_best_model_at_end' in _filtered_kwargs:
            _filtered_kwargs['load_best_model_at_end'] = False
            _filtered_kwargs.pop('metric_for_best_model', None)
            _filtered_kwargs.pop('greater_is_better', None)

        training_args = TrainingArguments(**_filtered_kwargs)
        
        # Initialize trainer with safer callback handling
        callbacks = None
        try:
            if (val_dataset and
                hasattr(training_args, "metric_for_best_model") and
                training_args.metric_for_best_model and
                hasattr(training_args, "evaluation_strategy") and
                training_args.evaluation_strategy != "no"):
                callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
                logger.info("Added EarlyStoppingCallback")
        except Exception as e:
            logger.warning(f"Could not add EarlyStoppingCallback: {e}. Training will continue without early stopping.")

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics if val_dataset else None,
            callbacks=callbacks,
        )
        
        # Train the model
        train_results = self.trainer.train()
        self.is_trained = True
        
        results.update({
            'train_results': train_results,
            'model': self.model,
            'trainer': self.trainer
        })
        
        # Evaluate on validation set if provided
        if val_dataset is not None:
            eval_results = self.trainer.evaluate()
            results['val_results'] = eval_results
            
            logger.info(f"CamemBERT training completed - Val Acc: {eval_results['eval_accuracy']:.4f}")
        else:
            logger.info("CamemBERT training completed")
        
        return results

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on new texts.

        Args:
            texts: List of texts to predict on

        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Create dataset
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        dataset = ThematicDataset(texts, dummy_labels, self.tokenizer)

        # Make predictions
        predictions = self.trainer.predict(dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        return predicted_labels

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            texts: List of texts to predict on

        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Create dataset
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        dataset = ThematicDataset(texts, dummy_labels, self.tokenizer)

        # Make predictions
        predictions = self.trainer.predict(dataset)
        probabilities = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

        return probabilities

    def evaluate(self, test_texts: List[str], test_labels: List[int],
                class_names: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_texts: Test texts
            test_labels: Test labels
            class_names: Names of classes for reporting

        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        logger.info("Evaluating CamemBERT model...")

        # Create test dataset
        test_dataset = ThematicDataset(test_texts, test_labels, self.tokenizer)

        # Evaluate
        eval_results = self.trainer.evaluate(test_dataset)

        # Make predictions for detailed metrics
        y_pred = self.predict(test_texts)
        y_proba = self.predict_proba(test_texts)

        # Calculate detailed metrics
        accuracy = accuracy_score(test_labels, y_pred)
        conf_matrix = confusion_matrix(test_labels, y_pred)
        class_report = classification_report(test_labels, y_pred, target_names=class_names, output_dict=True)

        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_proba,
            'eval_results': eval_results
        }

        logger.info(f"CamemBERT test accuracy: {accuracy:.4f}")

        return results

    def save_model(self, filepath: str):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Save model and tokenizer
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)

        # Save additional metadata
        import json
        metadata = {
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'best_params': self.best_params
        }

        with open(f"{filepath}/metadata.json", 'w') as f:
            json.dump(metadata, f)

        logger.info(f"CamemBERT model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        # Load metadata
        import json
        with open(f"{filepath}/metadata.json", 'r') as f:
            metadata = json.load(f)

        self.num_classes = metadata['num_classes']
        self.model_name = metadata['model_name']
        self.best_params = metadata['best_params']

        # Load model and tokenizer
        self.model = CamembertForSequenceClassification.from_pretrained(filepath)
        self.tokenizer = CamembertTokenizer.from_pretrained(filepath)

        # Create a dummy trainer for prediction
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir='./tmp',
            per_device_eval_batch_size=16,
            remove_unused_columns=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics
        )

        self.is_trained = True

        logger.info(f"CamemBERT model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("CamemBERT Thematic Classification Model - Example Usage")

    # This would typically be called with real data from the data preparation pipeline
    print("CamemBERT model implementation ready for integration with data preparation pipeline")
