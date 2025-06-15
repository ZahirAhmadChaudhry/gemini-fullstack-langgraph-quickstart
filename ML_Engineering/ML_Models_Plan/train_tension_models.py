"""
Comprehensive Tension Detection Model Training and Evaluation
============================================================

This script trains all 4 tension detection models, evaluates their performance,
and selects the best performing model for production deployment.

Models:
1. Random Forest Classifier
2. XGBoost Classifier  
3. SVM with RBF Kernel
4. Ensemble Voting Classifier

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
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from data_preparation import DataPreparationPipeline
from models.tension_detection.random_forest import TensionRandomForestModel
from models.tension_detection.xgboost_model import TensionXGBoostModel
from models.tension_detection.svm_model import TensionSVMModel
from models.tension_detection.ensemble_model import TensionEnsembleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tension_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TensionModelTrainer:
    """
    Comprehensive trainer for all tension detection models.
    """
    
    def __init__(self, data_dir: str = "data_from_Data_Engineering", 
                 output_dir: str = "trained_models"):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory containing the data engineering outputs
            output_dir: Directory to save trained models
        """
        # Resolve data directory to an absolute path to avoid issues when the
        # script is executed from a sub-directory.
        provided_path = Path(data_dir)

        if provided_path.exists():
            self.data_dir = str(provided_path)
        else:
            fallback_path = Path(__file__).resolve().parent.parent / data_dir
            if fallback_path.exists():
                self.data_dir = str(fallback_path)
                logger.warning(
                    "Provided data_dir '%s' not found. Falling back to '%s'.",
                    data_dir, self.data_dir
                )
            else:
                raise FileNotFoundError(
                    f"Data directory '{data_dir}' not found. Tried '{provided_path.resolve()}' and "
                    f"'{fallback_path}'. Please verify the path or pass the correct absolute path."
                )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data pipeline
        self.data_pipeline = DataPreparationPipeline(self.data_dir)
        self.results = {}
        
        logger.info(f"Initialized TensionModelTrainer with data_dir: {self.data_dir}")
    
    def prepare_data(self) -> dict:
        """
        Prepare data for training.
        
        Returns:
            Dictionary with prepared datasets and splits
        """
        logger.info("Preparing data for tension detection models...")
        
        try:
            # Run complete data preparation
            data_results = self.data_pipeline.run_complete_preparation()
            
            # Extract tension-specific data
            tension_dataset = data_results['datasets']['tension']
            tension_splits = data_results['splits']['tension']
            
            logger.info(f"Data preparation completed:")
            logger.info(f"  Total samples: {len(tension_dataset)}")
            logger.info(f"  Training samples: {len(tension_splits['X_train'])}")
            logger.info(f"  Validation samples: {len(tension_splits['X_val'])}")
            logger.info(f"  Test samples: {len(tension_splits['X_test'])}")
            logger.info(f"  Class distribution: {tension_splits['class_distribution']}")
            
            return data_results
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def train_all_models(self, data_results: dict) -> dict:
        """
        Train all tension detection models.
        
        Args:
            data_results: Results from data preparation
            
        Returns:
            Dictionary with training results for all models
        """
        logger.info("Starting comprehensive tension detection model training...")
        
        # Extract data
        splits = data_results['splits']['tension']
        class_weights = data_results['class_weights']['tension']
        tension_encoder = data_results['encoders']['tension']
        
        X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
        y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']
        feature_columns = splits['feature_columns']
        class_names = list(tension_encoder.classes_)
        
        models_results = {}
        
        # Model configurations
        model_configs = [
            {
                'name': 'Random Forest',
                'class': TensionRandomForestModel,
                'trials': 50,
                'filename': 'tension_random_forest_full.joblib'
            },
            {
                'name': 'XGBoost',
                'class': TensionXGBoostModel,
                'trials': 50,
                'filename': 'tension_xgboost_full.joblib'
            },
            {
                'name': 'SVM',
                'class': TensionSVMModel,
                'trials': 30,  # Reduced for SVM due to computational cost
                'filename': 'tension_svm_full.joblib'
            },
            {
                'name': 'Ensemble',
                'class': TensionEnsembleModel,
                'trials': 20,  # Reduced for ensemble due to complexity
                'filename': 'tension_ensemble_full.joblib'
            }
        ]
        
        # Train each model
        for config in model_configs:
            model_name = config['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name} Model")
            logger.info(f"{'='*60}")
            
            try:
                # Initialize model
                model = config['class']()
                
                # Train model
                training_results = model.train(
                    X_train, y_train, X_val, y_val,
                    class_weights=class_weights,
                    optimize=True,
                    n_trials=config['trials']
                )
                
                # Evaluate on test set
                evaluation_results = model.evaluate(
                    X_test, y_test,
                    class_names=class_names
                )
                
                # Save model
                model_path = self.output_dir / config['filename']
                model.save_model(str(model_path))
                
                # Store results
                models_results[model_name.lower().replace(' ', '_')] = {
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
                models_results[model_name.lower().replace(' ', '_')] = {
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
        logger.info("MODEL COMPARISON AND SELECTION")
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
                    'Status': 'Success'
                })
            else:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Test_Accuracy': 0,
                    'Val_Accuracy': 0,
                    'Macro_F1': 0,
                    'Weighted_F1': 0,
                    'Status': 'Failed'
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best model
        successful_models = comparison_df[comparison_df['Status'] == 'Success']
        
        if len(successful_models) > 0:
            # Select best model based on test accuracy
            best_model_idx = successful_models['Test_Accuracy'].idxmax()
            best_model_name = successful_models.loc[best_model_idx, 'Model']
            best_model_accuracy = successful_models.loc[best_model_idx, 'Test_Accuracy']
            
            logger.info("Model Performance Comparison:")
            logger.info(f"\n{comparison_df.to_string(index=False)}")
            
            logger.info(f"\n🏆 BEST MODEL SELECTED: {best_model_name}")
            logger.info(f"🎯 Test Accuracy: {best_model_accuracy:.4f}")
            
            # Check if target is met
            target_min = 0.75  # 75% minimum target
            target_max = 0.90  # 90% maximum target
            
            if best_model_accuracy >= target_min:
                if best_model_accuracy <= target_max:
                    status = "✅ TARGET ACHIEVED"
                else:
                    status = "🚀 TARGET EXCEEDED"
                logger.info(f"📊 Performance Status: {status}")
            else:
                logger.info(f"⚠️ Performance Status: BELOW TARGET (Target: {target_min:.1%}+)")
            
        else:
            logger.error("❌ No models trained successfully!")
            best_model_name = None
            best_model_accuracy = 0
        
        return {
            'comparison_df': comparison_df,
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
        logger.info("Saving comprehensive training results...")
        
        # Create summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_statistics': {
                'total_samples': len(data_results['datasets']['tension']),
                'train_samples': len(data_results['splits']['tension']['X_train']),
                'val_samples': len(data_results['splits']['tension']['X_val']),
                'test_samples': len(data_results['splits']['tension']['X_test']),
                'class_distribution': data_results['splits']['tension']['class_distribution'],
                'feature_count': len(data_results['splits']['tension']['feature_columns'])
            },
            'model_performance': comparison_results['comparison_df'].to_dict('records'),
            'best_model': {
                'name': comparison_results['best_model'],
                'accuracy': comparison_results['best_accuracy'],
                'target_achieved': comparison_results['target_achieved']
            },
            'target_metrics': {
                'original_accuracy': 0.33,  # 33% baseline
                'target_range': [0.75, 0.90],  # 75-90% target
                'improvement': comparison_results['best_accuracy'] - 0.33 if comparison_results['best_model'] else 0
            }
        }
        
        # Save summary (convert any NumPy scalars to native Python so json.dump works)
        def _json_default(obj):
            import numpy as _np
            if isinstance(obj, _np.generic):
                return obj.item()
            return str(obj)

        summary_path = self.output_dir / "tension_models_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_json_default)
        
        # -----------------------------------------------------------
        # Visualization section
        # -----------------------------------------------------------
        try:
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # 1) Bar chart of test accuracies
            acc_plot_path = plots_dir / "model_test_accuracies.png"
            plt.figure(figsize=(8,4))
            sns.barplot(
                x=comparison_results['comparison_df']['Model'],
                y=comparison_results['comparison_df']['Test_Accuracy']
            )
            plt.title('Test Accuracy per Model')
            plt.ylim(0,1)
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(acc_plot_path)
            plt.close()

            # 2) Confusion matrix heatmap for best model
            best_model_key = comparison_results['best_model'].lower().replace(' ', '_') if comparison_results['best_model'] else None
            if best_model_key and best_model_key in models_results and 'evaluation' in models_results[best_model_key]:
                conf = models_results[best_model_key]['evaluation']['confusion_matrix']
                cm_plot_path = plots_dir / f"{best_model_key}_confusion_matrix.png"
                plt.figure(figsize=(6,5))
                sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
                best_title = f"Confusion Matrix - {comparison_results['best_model']}"
                plt.title(best_title)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig(cm_plot_path)
                plt.close()

            logger.info(f"📊 Plots saved in {plots_dir}")
        except Exception as viz_err:
            logger.warning(f"Plot generation failed: {viz_err}")
        
        return summary
    
    def run_complete_training(self):
        """
        Run the complete tension detection model training pipeline.
        """
        logger.info("🚀 Starting Complete Tension Detection Model Training Pipeline")
        logger.info("="*80)
        
        try:
            # Step 1: Prepare data
            data_results = self.prepare_data()
            
            # Step 2: Train all models
            models_results = self.train_all_models(data_results)
            
            # Step 3: Compare models and select best
            comparison_results = self.compare_models(models_results)
            
            # Step 4: Save results
            summary = self.save_results(data_results, models_results, comparison_results)
            
            logger.info("\n" + "="*80)
            logger.info("🎉 TENSION DETECTION MODEL TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            
            return {
                'data': data_results,
                'models': models_results,
                'comparison': comparison_results,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Run the complete training pipeline
    trainer = TensionModelTrainer()
    results = trainer.run_complete_training()
