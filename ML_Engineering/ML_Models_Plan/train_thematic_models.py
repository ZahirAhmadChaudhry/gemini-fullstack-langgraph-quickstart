"""
Comprehensive Thematic Classification Model Training and Evaluation
==================================================================

This script trains all 4 thematic classification models, evaluates their performance,
and selects the best performing model for production deployment.

Models:
1. CamemBERT (French Transformer)
2. Logistic Regression with TF-IDF
3. Sentence-BERT + Neural Network
4. Naive Bayes with Feature Engineering

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
import torch
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from data_preparation import DataPreparationPipeline
from models.thematic_classification.camembert_model import ThematicCamemBERTModel
from models.thematic_classification.logistic_regression import ThematicLogisticRegressionModel
from models.thematic_classification.sentence_bert import ThematicSentenceBERTModel
from models.thematic_classification.naive_bayes import ThematicNaiveBayesModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thematic_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# ThematicModelTrainer
# ----------------------------------------------------------------------------

class ThematicModelTrainer:
    """
    Comprehensive trainer for all thematic classification models.
    """
    
    def __init__(self, data_dir: str = "data_from_Data_Engineering", 
                 output_dir: str = "trained_models"):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory containing the data engineering outputs
            output_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data pipeline
        self.data_pipeline = DataPreparationPipeline(data_dir)
        self.results = {}
        
        logger.info(f"Initialized ThematicModelTrainer with data_dir: {data_dir}")
    
    def prepare_data(self, table_name: str | None = None) -> dict:
        """
        Prepare data for training.
        
        Returns:
            Dictionary with prepared datasets and splits
        """
        logger.info("Preparing data for thematic classification models...")
        
        try:
            if table_name:
                # Quick single-table path for smoke testing
                logger.info(f"⚡ Loading SINGLE TABLE data: {table_name} …")
                target_df, ml_df = self.data_pipeline.load_single_table_data(table_name)

                # Prepare thematic dataset and create splits manually
                thematic_dataset = self.data_pipeline.prepare_thematic_dataset(target_df, ml_df)

                if thematic_dataset.empty:
                    raise ValueError("Single-table dataset produced zero samples. Check data alignment.")

                thematic_splits = self.data_pipeline.create_train_test_splits(
                    thematic_dataset, 'theme_label')

                data_results = {
                    'datasets': {'thematic': thematic_dataset},
                    'splits': {'thematic': thematic_splits},
                    'class_weights': {'thematic': self.data_pipeline.get_class_weights(thematic_splits['y_train'])},
                    'encoders': {'thematic': self.data_pipeline.theme_encoder}
                }
            else:
                # Full dataset preparation
                data_results = self.data_pipeline.run_complete_preparation()
            
            # Extract thematic-specific data
            thematic_dataset = data_results['datasets']['thematic']
            thematic_splits = data_results['splits']['thematic']
            
            logger.info(f"Data preparation completed:")
            logger.info(f"  Total samples: {len(thematic_dataset)}")
            logger.info(f"  Training samples: {len(thematic_splits['train_indices'])}")
            logger.info(f"  Validation samples: {len(thematic_splits['val_indices'])}")
            logger.info(f"  Test samples: {len(thematic_splits['test_indices'])}")
            logger.info(f"  Class distribution: {thematic_splits['class_distribution']}")
            
            return data_results
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def train_all_models(self, data_results: dict, include_deep: bool = False) -> dict:
        """
        Train all thematic classification models.
        
        Args:
            data_results: Results from data preparation
            include_deep: Whether to include deep transformer models
            
        Returns:
            Dictionary with training results for all models
        """
        logger.info("Starting comprehensive thematic classification model training...")
        
        # Extract data
        thematic_dataset = data_results['datasets']['thematic']
        splits = data_results['splits']['thematic']
        class_weights = data_results['class_weights']['thematic']
        thematic_encoder = data_results['encoders']['thematic']
        
        # Get text data and labels
        train_indices = splits['train_indices']
        val_indices = splits['val_indices']
        test_indices = splits['test_indices']
        
        train_texts = thematic_dataset.iloc[train_indices]['text'].tolist()
        val_texts = thematic_dataset.iloc[val_indices]['text'].tolist()
        test_texts = thematic_dataset.iloc[test_indices]['text'].tolist()
        
        train_labels = splits['y_train'].tolist()
        val_labels = splits['y_val'].tolist()
        test_labels = splits['y_test'].tolist()
        
        class_names = list(thematic_encoder.classes_)
        
        models_results = {}
        
        # Model configurations
        from models.thematic_classification import (
            ThematicLogisticRegressionModel,
            ThematicNaiveBayesModel,
            ThematicMiniLMSVMModel,
        )

        models_to_train = {
            'Logistic Regression': ThematicLogisticRegressionModel(),
            'Naive Bayes': ThematicNaiveBayesModel(),
            'MiniLM SVM': ThematicMiniLMSVMModel(),
        }

        if include_deep:
            # Check system constraints before loading deep models
            import psutil
            import platform

            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            is_windows = platform.system() == "Windows"

            if is_windows and available_memory_gb < 8:
                logger.warning(f"Insufficient memory ({available_memory_gb:.1f}GB) for deep models on Windows")
                logger.warning("Skipping deep models to prevent system instability")
            else:
                try:
                    from models.thematic_classification import (
                        ThematicSentenceBERTModel,
                        ThematicCamemBERTModel,
                    )

                    # Add Sentence-BERT with error handling
                    try:
                        models_to_train['Sentence BERT'] = ThematicSentenceBERTModel()
                        logger.info("Successfully loaded Sentence-BERT model")
                    except Exception as e:
                        logger.error(f"Failed to load Sentence-BERT model: {e}")
                        logger.warning("Skipping Sentence-BERT model")

                    # Add CamemBERT with error handling
                    try:
                        models_to_train['CamemBERT'] = ThematicCamemBERTModel(num_classes=2)
                        logger.info("Successfully loaded CamemBERT model")
                    except Exception as e:
                        logger.error(f"Failed to load CamemBERT model: {e}")
                        logger.warning("Skipping CamemBERT model")

                except ImportError as e:
                    logger.error(f"Failed to import deep models: {e}")
                    logger.warning("Deep models not available - continuing with light models only")
        
        # Train each model
        for model_name, model in models_to_train.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name} Model")
            logger.info(f"{'='*60}")
            
            try:
                # Prepare class weights
                if model_name in ['Sentence-BERT', 'CamemBERT']:
                    # Convert to torch tensor for neural models
                    class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)
                    model_class_weights = class_weights_tensor
                else:
                    model_class_weights = class_weights
                
                # Train model
                training_results = model.train(
                    train_texts, train_labels, val_texts, val_labels,
                    class_weights=model_class_weights,
                    optimize=True,
                    n_trials=50
                )
                
                # Evaluate on test set
                evaluation_results = model.evaluate(
                    test_texts, test_labels,
                    class_names=class_names
                )
                
                # Save model
                model_path = self.output_dir / f"{model_name.lower().replace(' ', '_').replace('-', '_')}_full.joblib"
                model.save_model(str(model_path))
                
                # Store results
                models_results[model_name.lower().replace(' ', '_').replace('-', '_')] = {
                    'training': training_results,
                    'evaluation': evaluation_results,
                    'model': model,
                    'model_path': str(model_path)
                }
                
                # Log performance
                test_accuracy = evaluation_results['accuracy']
                val_accuracy = training_results.get('val_accuracy', 'N/A')
                
                logger.info(f"{model_name} Results:")
                logger.info(f"  Validation Accuracy: {val_accuracy}")
                logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
                logger.info(f"  Model saved to: {model_path}")
                
            except Exception as e:
                logger.error(f"{model_name} training failed: {str(e)}")
                models_results[model_name.lower().replace(' ', '_').replace('-', '_')] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return models_results
    
    def compare_models(self, models_results: dict) -> dict:
        """
        Compare all trained models and select the best one.
        
        Args:
            models_results: Results from model training
            
        Returns:
            Dictionary with model comparison and best model selection
        """
        logger.info("\n" + "="*60)
        logger.info("THEMATIC MODEL COMPARISON AND SELECTION")
        logger.info("="*60)
        
        # Extract performance metrics
        comparison_data = []
        
        for model_name, results in models_results.items():
            if 'evaluation' in results:
                eval_results = results['evaluation']
                training_results = results['training']
                
                # Get class-wise metrics
                class_report = eval_results['classification_report']
                
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Test_Accuracy': eval_results['accuracy'],
                    'Val_Accuracy': training_results.get('val_accuracy', 0),
                    'Macro_F1': class_report['macro avg']['f1-score'],
                    'Weighted_F1': class_report['weighted avg']['f1-score'],
                    'Performance_F1': class_report.get('Performance', {}).get('f1-score', 0),
                    'Legitimacy_F1': class_report.get('Légitimité', {}).get('f1-score', 0),
                    'Status': 'Success'
                })
            else:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Test_Accuracy': 0,
                    'Val_Accuracy': 0,
                    'Macro_F1': 0,
                    'Weighted_F1': 0,
                    'Performance_F1': 0,
                    'Legitimacy_F1': 0,
                    'Status': 'Failed'
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank by macro-F1 primarily, then accuracy
        comparison_df['RankScore'] = comparison_df['Macro_F1'] * 0.7 + comparison_df['Test_Accuracy'] * 0.3
        
        # Find best model
        successful_models = comparison_df[comparison_df['Status'] == 'Success']
        
        if len(successful_models) > 0:
            # Select best model based on test accuracy
            best_model_idx = successful_models['RankScore'].idxmax()
            best_model_name = successful_models.loc[best_model_idx, 'Model']
            best_model_accuracy = successful_models.loc[best_model_idx, 'Test_Accuracy']
            
            logger.info("Thematic Model Performance Comparison:")
            logger.info(f"\n{comparison_df.to_string(index=False)}")
            
            logger.info(f"\n🏆 BEST THEMATIC MODEL SELECTED: {best_model_name}")
            logger.info(f"🎯 Test Accuracy: {best_model_accuracy:.4f}")
            
            # Check if target is met
            target_min = 0.85  # 85% minimum target
            target_max = 0.95  # 95% maximum target
            
            if best_model_accuracy >= target_min:
                if best_model_accuracy <= target_max:
                    status = "✅ TARGET ACHIEVED"
                else:
                    status = "🚀 TARGET EXCEEDED"
                logger.info(f"📊 Performance Status: {status}")
            else:
                logger.info(f"⚠️ Performance Status: BELOW TARGET (Target: {target_min:.1%}+)")
            
        else:
            logger.error("❌ No thematic models trained successfully!")
            best_model_name = None
            best_model_accuracy = 0
        
        return {
            'comparison_df': comparison_df.drop(columns=['RankScore']),
            'best_model': best_model_name,
            'best_accuracy': best_model_accuracy,
            'target_achieved': best_model_accuracy >= target_min if best_model_name else False
        }
    
    def save_results(self, data_results: dict, models_results: dict, comparison_results: dict):
        """
        Save comprehensive training results.
        
        Args:
            data_results: Data preparation results
            models_results: Model training results
            comparison_results: Model comparison results
        """
        logger.info("Saving comprehensive thematic training results...")
        
        # Create summary report
        import numpy as _np

        def _json_default(obj):
            """Convert objects not serializable by default json encoder."""
            import numpy as _np
            import pandas as _pd
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, (_np.ndarray, list, tuple)):
                return obj.tolist() if hasattr(obj, "tolist") else list(obj)
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            if isinstance(obj, _pd.DataFrame):
                return obj.to_dict()
            if isinstance(obj, _pd.Series):
                return obj.to_dict()
            return str(obj)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_statistics': {
                'total_samples': len(data_results['datasets']['thematic']),
                'train_samples': len(data_results['splits']['thematic']['train_indices']),
                'val_samples': len(data_results['splits']['thematic']['val_indices']),
                'test_samples': len(data_results['splits']['thematic']['test_indices']),
                'class_distribution': (
                    data_results['splits']['thematic']['class_distribution'].tolist()
                    if isinstance(data_results['splits']['thematic']['class_distribution'], _np.ndarray)
                    else data_results['splits']['thematic']['class_distribution']
                )
            },
            'model_performance': comparison_results['comparison_df'].to_dict('records'),
            'best_model': {
                'name': comparison_results['best_model'],
                'accuracy': comparison_results['best_accuracy'],
                'target_achieved': comparison_results['target_achieved']
            },
            'target_metrics': {
                'target_range': [0.85, 0.95],  # 85-95% target
                'class_balance_challenge': 'Performance: 89.4%, Légitimité: 10.6%'
            }
        }
        
        # Save summary
        summary_path = self.output_dir / "thematic_models_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_json_default)
        
        # Save comparison DataFrame
        comparison_path = self.output_dir / "thematic_models_comparison.csv"
        comparison_results['comparison_df'].to_csv(comparison_path, index=False)
        
        logger.info(f"Thematic results saved to:")
        logger.info(f"  Summary: {summary_path}")
        logger.info(f"  Comparison: {comparison_path}")
        
        return summary
    
    def run_complete_training(self, table_name: str | None = None, include_deep: bool = False):
        """
        Run the complete thematic classification model training pipeline.
        """
        logger.info("🚀 Starting Complete Thematic Classification Model Training Pipeline")
        logger.info("="*80)
        
        try:
            # Step 1: Prepare data
            data_results = self.prepare_data(table_name)
            
            # Step 2: Train all models
            models_results = self.train_all_models(data_results, include_deep=include_deep)
            
            # Step 3: Compare models and select best
            comparison_results = self.compare_models(models_results)
            
            # Step 4: Save results
            summary = self.save_results(data_results, models_results, comparison_results)
            
            logger.info("\n" + "="*80)
            logger.info("🎉 THEMATIC CLASSIFICATION MODEL TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            
            return {
                'data': data_results,
                'models': models_results,
                'comparison': comparison_results,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Thematic training pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train thematic classification models")
    parser.add_argument("--table", help="Single table name (e.g., Table_A) for quick testing", default=None)
    parser.add_argument("--deep", action="store_true", help="Include heavy transformer fine-tuning models (CamemBERT, SBERT)")
    args = parser.parse_args()

    trainer = ThematicModelTrainer()
    results = trainer.run_complete_training(table_name=args.table, include_deep=args.deep)
