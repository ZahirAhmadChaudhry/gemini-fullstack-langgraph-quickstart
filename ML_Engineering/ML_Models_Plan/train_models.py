"""
Main Training Pipeline for ML Models
====================================

Orchestrates the complete training pipeline for tension detection and thematic classification models.

Author: ML Engineering Team
Date: 2025-06-12
"""

import sys
import os
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from data_preparation import DataPreparationPipeline
from models.tension_detection.random_forest import TensionRandomForestModel
from models.tension_detection.xgboost_model import TensionXGBoostModel
from models.thematic_classification.camembert_model import ThematicCamemBERTModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """
    Main training pipeline that orchestrates data preparation and model training.
    """
    
    def __init__(self, data_dir: str = "data_from_Data_Engineering", 
                 output_dir: str = "trained_models"):
        """
        Initialize the training pipeline.
        
        Args:
            data_dir: Directory containing the data engineering outputs
            output_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_pipeline = DataPreparationPipeline(data_dir)
        self.results = {}
        
        logger.info(f"Initialized MLTrainingPipeline with data_dir: {data_dir}")
    
    def prepare_data(self) -> dict:
        """
        Run the complete data preparation pipeline.
        
        Returns:
            Dictionary with prepared datasets and splits
        """
        logger.info("Starting data preparation...")
        
        try:
            # Run complete data preparation
            data_results = self.data_pipeline.run_complete_preparation()
            
            # Log data statistics
            tension_dataset = data_results['datasets']['tension']
            thematic_dataset = data_results['datasets']['thematic']
            
            logger.info(f"Data preparation completed:")
            logger.info(f"  Tension dataset: {len(tension_dataset)} samples")
            logger.info(f"  Thematic dataset: {len(thematic_dataset)} samples")
            
            # Log class distributions
            tension_splits = data_results['splits']['tension']
            thematic_splits = data_results['splits']['thematic']
            
            logger.info(f"  Tension class distribution: {tension_splits['class_distribution']}")
            logger.info(f"  Thematic class distribution: {thematic_splits['class_distribution']}")
            
            return data_results
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def train_tension_models(self, data_results: dict) -> dict:
        """
        Train all tension detection models.
        
        Args:
            data_results: Results from data preparation
            
        Returns:
            Dictionary with training results for all tension models
        """
        logger.info("Starting tension detection model training...")
        
        # Extract data
        splits = data_results['splits']['tension']
        class_weights = data_results['class_weights']['tension']
        tension_encoder = data_results['encoders']['tension']
        
        X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
        y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']
        feature_columns = splits['feature_columns']
        
        tension_results = {}
        
        # Train Random Forest
        logger.info("Training Random Forest for tension detection...")
        try:
            rf_model = TensionRandomForestModel()
            rf_results = rf_model.train(
                X_train, y_train, X_val, y_val, 
                class_weights=class_weights, 
                optimize=True, n_trials=50
            )
            
            # Evaluate on test set
            rf_eval = rf_model.evaluate(
                X_test, y_test, 
                class_names=list(tension_encoder.classes_)
            )
            
            # Save model
            rf_model.save_model(self.output_dir / "tension_random_forest.joblib")
            
            tension_results['random_forest'] = {
                'training': rf_results,
                'evaluation': rf_eval,
                'model': rf_model
            }
            
            logger.info(f"Random Forest completed - Test Accuracy: {rf_eval['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}")
            tension_results['random_forest'] = {'error': str(e)}
        
        # Train XGBoost
        logger.info("Training XGBoost for tension detection...")
        try:
            xgb_model = TensionXGBoostModel()
            xgb_results = xgb_model.train(
                X_train, y_train, X_val, y_val,
                class_weights=class_weights,
                optimize=True, n_trials=50
            )
            
            # Evaluate on test set
            xgb_eval = xgb_model.evaluate(
                X_test, y_test,
                class_names=list(tension_encoder.classes_)
            )
            
            # Save model
            xgb_model.save_model(self.output_dir / "tension_xgboost.joblib")
            
            tension_results['xgboost'] = {
                'training': xgb_results,
                'evaluation': xgb_eval,
                'model': xgb_model
            }
            
            logger.info(f"XGBoost completed - Test Accuracy: {xgb_eval['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}")
            tension_results['xgboost'] = {'error': str(e)}
        
        return tension_results
    
    def train_thematic_models(self, data_results: dict) -> dict:
        """
        Train all thematic classification models.
        
        Args:
            data_results: Results from data preparation
            
        Returns:
            Dictionary with training results for all thematic models
        """
        logger.info("Starting thematic classification model training...")
        
        # Extract data
        thematic_dataset = data_results['datasets']['thematic']
        splits = data_results['splits']['thematic']
        class_weights = data_results['class_weights']['thematic']
        thematic_encoder = data_results['encoders']['thematic']
        
        # Get text data for transformer models
        train_indices = splits['train_indices']
        val_indices = splits['val_indices']
        test_indices = splits['test_indices']
        
        train_texts = thematic_dataset.iloc[train_indices]['text'].tolist()
        val_texts = thematic_dataset.iloc[val_indices]['text'].tolist()
        test_texts = thematic_dataset.iloc[test_indices]['text'].tolist()
        
        train_labels = splits['y_train'].tolist()
        val_labels = splits['y_val'].tolist()
        test_labels = splits['y_test'].tolist()
        
        thematic_results = {}
        
        # Train CamemBERT (reduced trials due to computational cost)
        logger.info("Training CamemBERT for thematic classification...")
        try:
            # Convert class weights to torch tensor
            import torch
            class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)
            
            camembert_model = ThematicCamemBERTModel(num_classes=len(thematic_encoder.classes_))
            camembert_results = camembert_model.train(
                train_texts, train_labels, val_texts, val_labels,
                class_weights=class_weights_tensor,
                optimize=True, n_trials=10  # Reduced for computational efficiency
            )
            
            # Evaluate on test set
            camembert_eval = camembert_model.evaluate(
                test_texts, test_labels,
                class_names=list(thematic_encoder.classes_)
            )
            
            # Save model
            camembert_model.save_model(str(self.output_dir / "thematic_camembert"))
            
            thematic_results['camembert'] = {
                'training': camembert_results,
                'evaluation': camembert_eval,
                'model': camembert_model
            }
            
            logger.info(f"CamemBERT completed - Test Accuracy: {camembert_eval['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"CamemBERT training failed: {str(e)}")
            thematic_results['camembert'] = {'error': str(e)}
        
        return thematic_results
    
    def save_results(self, data_results: dict, tension_results: dict, thematic_results: dict):
        """
        Save training results and generate summary report.
        
        Args:
            data_results: Data preparation results
            tension_results: Tension model training results
            thematic_results: Thematic model training results
        """
        logger.info("Saving training results...")
        
        # Compile summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_statistics': {
                'tension_samples': len(data_results['datasets']['tension']),
                'thematic_samples': len(data_results['datasets']['thematic']),
                'tension_classes': list(data_results['encoders']['tension'].classes_),
                'thematic_classes': list(data_results['encoders']['thematic'].classes_)
            },
            'model_performance': {}
        }
        
        # Add tension model results
        for model_name, results in tension_results.items():
            if 'evaluation' in results:
                summary['model_performance'][f'tension_{model_name}'] = {
                    'accuracy': results['evaluation']['accuracy'],
                    'status': 'success'
                }
            else:
                summary['model_performance'][f'tension_{model_name}'] = {
                    'status': 'failed',
                    'error': results.get('error', 'Unknown error')
                }
        
        # Add thematic model results
        for model_name, results in thematic_results.items():
            if 'evaluation' in results:
                summary['model_performance'][f'thematic_{model_name}'] = {
                    'accuracy': results['evaluation']['accuracy'],
                    'status': 'success'
                }
            else:
                summary['model_performance'][f'thematic_{model_name}'] = {
                    'status': 'failed',
                    'error': results.get('error', 'Unknown error')
                }
        
        # Save summary
        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training results saved to {self.output_dir}")
        logger.info("Training Summary:")
        for model, perf in summary['model_performance'].items():
            if perf['status'] == 'success':
                logger.info(f"  {model}: {perf['accuracy']:.4f} accuracy")
            else:
                logger.info(f"  {model}: FAILED - {perf.get('error', 'Unknown error')}")
    
    def run_complete_training(self):
        """
        Run the complete training pipeline.
        """
        logger.info("Starting complete ML training pipeline...")
        
        try:
            # Step 1: Prepare data
            data_results = self.prepare_data()
            
            # Step 2: Train tension detection models
            tension_results = self.train_tension_models(data_results)
            
            # Step 3: Train thematic classification models
            thematic_results = self.train_thematic_models(data_results)
            
            # Step 4: Save results
            self.save_results(data_results, tension_results, thematic_results)
            
            logger.info("Complete ML training pipeline finished successfully!")
            
            return {
                'data': data_results,
                'tension': tension_results,
                'thematic': thematic_results
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Run the complete training pipeline
    pipeline = MLTrainingPipeline()
    results = pipeline.run_complete_training()
