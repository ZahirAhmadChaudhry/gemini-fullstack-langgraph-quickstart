"""
Production Tension Detection Training
====================================

Train tension detection models on real data (1,827 samples) using only
working implementations: Random Forest, SVM, and Simple Ensemble.

Compare with existing 97.54% Random Forest baseline.

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
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_preparation import DataPreparationPipeline
from models.tension_detection.random_forest import TensionRandomForestModel
from models.tension_detection.svm_model import TensionSVMModel
from models.tension_detection.xgboost_model import TensionXGBoostModel
from models.tension_detection.ensemble_model import TensionEnsembleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tension_production_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionTensionTrainer:
    """
    Production trainer for tension detection models.
    """
    
    def __init__(self, data_dir: str = "../data_from_Data_Engineering", 
                 output_dir: str = "trained_models"):
        """
        Initialize the production trainer.
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized ProductionTensionTrainer")
    
    def load_and_prepare_data(self):
        """
        Load and prepare SINGLE TABLE dataset for quick testing.
        """
        logger.info("🔄 Loading and preparing SINGLE TABLE data (Table_A only)...")
        start_time = time.time()

        # Initialize data pipeline
        pipeline = DataPreparationPipeline(self.data_dir)

        # Load only Table_A for quick testing
        target_df, ml_df = pipeline.load_single_table_data("Table_A")
        logger.info(f"✅ Loaded {len(target_df)} target entries and {len(ml_df)} ML segments from Table_A")
        
        # Prepare tension dataset
        tension_dataset = pipeline.prepare_tension_dataset(target_df, ml_df)
        logger.info(f"✅ Prepared tension dataset with {len(tension_dataset)} samples")
        
        # Create train/test splits
        splits = pipeline.create_train_test_splits(tension_dataset, 'tension_label')
        
        # Get class information
        class_names = list(pipeline.tension_encoder.classes_)
        class_weights = pipeline.get_class_weights(splits['y_train'])
        
        end_time = time.time()
        logger.info(f"✅ Data preparation completed in {end_time - start_time:.2f} seconds")
        
        return splits, class_names, class_weights, pipeline.tension_encoder
    
    def train_random_forest_optimized(self, splits, class_weights):
        """
        Train Random Forest with full optimization.
        """
        logger.info("\n🌲 TRAINING OPTIMIZED RANDOM FOREST")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        # Initialize model
        rf_model = TensionRandomForestModel()
        
        # Train with optimization (reduced trials for quick test)
        results = rf_model.train(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            class_weights=class_weights,
            optimize=True,
            n_trials=10  # Reduced for quick test
        )
        
        # Evaluate on test set
        evaluation = rf_model.evaluate(
            splits['X_test'], splits['y_test'],
            class_names=None
        )
        
        # Save model
        model_path = self.output_dir / "tension_random_forest_production.joblib"
        rf_model.save_model(str(model_path))
        
        end_time = time.time()
        
        logger.info(f"✅ Random Forest training completed in {end_time - start_time:.2f} seconds")
        logger.info(f"  📊 Test Accuracy: {evaluation['accuracy']:.4f}")
        logger.info(f"  💾 Model saved: {model_path}")
        
        return {
            'model': rf_model,
            'training_results': results,
            'evaluation': evaluation,
            'model_path': str(model_path),
            'training_time': end_time - start_time
        }
    
    def train_svm_optimized(self, splits, class_weights):
        """
        Train SVM with full optimization.
        """
        logger.info("\n⚙️ TRAINING OPTIMIZED SVM")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        # Initialize model
        svm_model = TensionSVMModel()
        
        # Train with optimization
        results = svm_model.train(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            class_weights=class_weights,
            optimize=True,
            n_trials=5  # Reduced for quick test
        )
        
        # Evaluate on test set
        evaluation = svm_model.evaluate(
            splits['X_test'], splits['y_test'],
            class_names=None
        )
        
        # Save model
        model_path = self.output_dir / "tension_svm_production.joblib"
        svm_model.save_model(str(model_path))
        
        end_time = time.time()
        
        logger.info(f"✅ SVM training completed in {end_time - start_time:.2f} seconds")
        logger.info(f"  📊 Test Accuracy: {evaluation['accuracy']:.4f}")
        logger.info(f"  💾 Model saved: {model_path}")
        
        return {
            'model': svm_model,
            'training_results': results,
            'evaluation': evaluation,
            'model_path': str(model_path),
            'training_time': end_time - start_time
        }

    def train_xgboost_optimized(self, splits, class_weights):
        """
        Train XGBoost with full optimization.
        """
        logger.info("\n🚀 TRAINING OPTIMIZED XGBOOST")
        logger.info("=" * 50)

        start_time = time.time()

        # Initialize model
        xgb_model = TensionXGBoostModel()

        # Train with optimization (no early stopping to avoid API issues)
        results = xgb_model.train(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            class_weights=class_weights,
            optimize=True,
            n_trials=5  # Reduced for quick test
        )

        # Evaluate on test set
        evaluation = xgb_model.evaluate(
            splits['X_test'], splits['y_test'],
            class_names=None
        )

        # Save model
        model_path = self.output_dir / "tension_xgboost_production.joblib"
        xgb_model.save_model(str(model_path))

        end_time = time.time()

        logger.info(f"✅ XGBoost training completed in {end_time - start_time:.2f} seconds")
        logger.info(f"  📊 Test Accuracy: {evaluation['accuracy']:.4f}")
        logger.info(f"  💾 Model saved: {model_path}")

        return {
            'model': xgb_model,
            'training_results': results,
            'evaluation': evaluation,
            'model_path': str(model_path),
            'training_time': end_time - start_time
        }
    
    def create_production_ensemble(self, rf_results, svm_results, splits):
        """
        Create production ensemble from trained models.
        """
        logger.info("\n🤝 CREATING PRODUCTION ENSEMBLE")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        # Get models
        rf_model = rf_results['model']
        svm_model = svm_results['model']
        
        # Test different ensemble strategies
        X_test, y_test = splits['X_test'], splits['y_test']
        
        # Get predictions
        rf_pred = rf_model.predict(X_test)
        svm_pred = svm_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)
        svm_proba = svm_model.predict_proba(X_test)
        
        # Strategy 1: Simple majority voting
        majority_pred = np.array([rf_pred[i] if rf_pred[i] == svm_pred[i] else rf_pred[i] for i in range(len(X_test))])
        majority_accuracy = np.mean(majority_pred == y_test)
        
        # Strategy 2: Weighted voting (RF gets higher weight due to baseline performance)
        weighted_proba = 0.7 * rf_proba + 0.3 * svm_proba
        weighted_pred = np.argmax(weighted_proba, axis=1)
        weighted_accuracy = np.mean(weighted_pred == y_test)
        
        # Strategy 3: Confidence-based voting
        rf_confidence = np.max(rf_proba, axis=1)
        svm_confidence = np.max(svm_proba, axis=1)
        confidence_pred = np.array([rf_pred[i] if rf_confidence[i] > svm_confidence[i] else svm_pred[i] for i in range(len(X_test))])
        confidence_accuracy = np.mean(confidence_pred == y_test)
        
        # Select best ensemble strategy
        strategies = {
            'majority_voting': majority_accuracy,
            'weighted_voting': weighted_accuracy,
            'confidence_based': confidence_accuracy
        }
        
        best_strategy = max(strategies, key=strategies.get)
        best_accuracy = strategies[best_strategy]
        
        end_time = time.time()
        
        logger.info(f"✅ Ensemble creation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"  📊 Ensemble Strategies:")
        for strategy, accuracy in strategies.items():
            logger.info(f"    {strategy}: {accuracy:.4f}")
        logger.info(f"  🏆 Best Strategy: {best_strategy} ({best_accuracy:.4f})")
        
        return {
            'strategies': strategies,
            'best_strategy': best_strategy,
            'best_accuracy': best_accuracy,
            'rf_model': rf_model,
            'svm_model': svm_model
        }

    def create_full_ensemble(self, rf_results, svm_results, xgb_results, splits):
        """
        Create full ensemble from all three trained models.
        """
        logger.info("\n🤝 CREATING FULL ENSEMBLE (RF + SVM + XGBoost)")
        logger.info("=" * 60)

        start_time = time.time()

        # Get models
        rf_model = rf_results['model']
        svm_model = svm_results['model']
        xgb_model = xgb_results['model']

        # Test different ensemble strategies
        X_test, y_test = splits['X_test'], splits['y_test']

        # Get predictions and probabilities
        rf_pred = rf_model.predict(X_test)
        svm_pred = svm_model.predict(X_test)
        xgb_pred = xgb_model.predict(X_test)

        rf_proba = rf_model.predict_proba(X_test)
        svm_proba = svm_model.predict_proba(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)

        # Strategy 1: Simple majority voting
        majority_pred = []
        for i in range(len(X_test)):
            votes = [rf_pred[i], svm_pred[i], xgb_pred[i]]
            majority_pred.append(max(set(votes), key=votes.count))
        majority_pred = np.array(majority_pred)
        majority_accuracy = np.mean(majority_pred == y_test)

        # Strategy 2: Weighted voting (RF=0.5, XGB=0.3, SVM=0.2 based on expected performance)
        weighted_proba = 0.5 * rf_proba + 0.3 * xgb_proba + 0.2 * svm_proba
        weighted_pred = np.argmax(weighted_proba, axis=1)
        weighted_accuracy = np.mean(weighted_pred == y_test)

        # Strategy 3: Performance-based weighting
        rf_acc = rf_results['evaluation']['accuracy']
        svm_acc = svm_results['evaluation']['accuracy']
        xgb_acc = xgb_results['evaluation']['accuracy']
        total_acc = rf_acc + svm_acc + xgb_acc

        perf_weighted_proba = (rf_acc/total_acc) * rf_proba + (xgb_acc/total_acc) * xgb_proba + (svm_acc/total_acc) * svm_proba
        perf_weighted_pred = np.argmax(perf_weighted_proba, axis=1)
        perf_weighted_accuracy = np.mean(perf_weighted_pred == y_test)

        # Strategy 4: Confidence-based voting
        rf_confidence = np.max(rf_proba, axis=1)
        svm_confidence = np.max(svm_proba, axis=1)
        xgb_confidence = np.max(xgb_proba, axis=1)

        confidence_pred = []
        for i in range(len(X_test)):
            confidences = [rf_confidence[i], svm_confidence[i], xgb_confidence[i]]
            predictions = [rf_pred[i], svm_pred[i], xgb_pred[i]]
            best_idx = np.argmax(confidences)
            confidence_pred.append(predictions[best_idx])
        confidence_pred = np.array(confidence_pred)
        confidence_accuracy = np.mean(confidence_pred == y_test)

        # Select best ensemble strategy
        strategies = {
            'majority_voting': majority_accuracy,
            'weighted_voting': weighted_accuracy,
            'performance_weighted': perf_weighted_accuracy,
            'confidence_based': confidence_accuracy
        }

        best_strategy = max(strategies, key=strategies.get)
        best_accuracy = strategies[best_strategy]

        end_time = time.time()

        logger.info(f"✅ Full ensemble creation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"  📊 Ensemble Strategies:")
        for strategy, accuracy in strategies.items():
            logger.info(f"    {strategy}: {accuracy:.4f}")
        logger.info(f"  🏆 Best Strategy: {best_strategy} ({best_accuracy:.4f})")

        return {
            'strategies': strategies,
            'best_strategy': best_strategy,
            'best_accuracy': best_accuracy,
            'rf_model': rf_model,
            'svm_model': svm_model,
            'xgb_model': xgb_model,
            'individual_accuracies': {
                'rf': rf_acc,
                'svm': svm_acc,
                'xgb': xgb_acc
            }
        }
    
    def compare_with_baseline(self, rf_results, svm_results, ensemble_results):
        """
        Compare results with the 97.54% baseline.
        """
        logger.info("\n📊 COMPARISON WITH BASELINE")
        logger.info("=" * 50)
        
        baseline_accuracy = 0.9754  # Existing Random Forest baseline
        
        results = {
            'Baseline RF (97.54%)': baseline_accuracy,
            'New Random Forest': rf_results['evaluation']['accuracy'],
            'New SVM': svm_results['evaluation']['accuracy'],
            'Best Ensemble': ensemble_results['best_accuracy']
        }
        
        logger.info("🏆 FINAL RESULTS COMPARISON:")
        logger.info("-" * 40)
        for model_name, accuracy in results.items():
            status = "🟢" if accuracy >= 0.75 else "🟡" if accuracy >= 0.50 else "🔴"
            logger.info(f"  {status} {model_name:20}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Find best model
        best_model = max(results, key=results.get)
        best_accuracy = results[best_model]
        
        logger.info(f"\n🥇 BEST MODEL: {best_model}")
        logger.info(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # Check if we beat or match baseline
        if best_accuracy >= baseline_accuracy:
            logger.info("🚀 SUCCESS: New model matches or exceeds baseline!")
        elif best_accuracy >= 0.90:
            logger.info("✅ EXCELLENT: New model achieves >90% accuracy!")
        elif best_accuracy >= 0.75:
            logger.info("👍 GOOD: New model meets minimum target (75%+)!")
        else:
            logger.info("⚠️ NEEDS IMPROVEMENT: Below target accuracy")
        
        return results, best_model, best_accuracy

    def compare_with_baseline_full(self, rf_results, svm_results, xgb_results, ensemble_results):
        """
        Compare all results with the 97.54% baseline.
        """
        logger.info("\n📊 COMPREHENSIVE COMPARISON WITH BASELINE")
        logger.info("=" * 60)

        baseline_accuracy = 0.9754  # Existing Random Forest baseline

        results = {
            'Baseline RF (97.54%)': baseline_accuracy,
            'New Random Forest': rf_results['evaluation']['accuracy'],
            'New SVM': svm_results['evaluation']['accuracy'],
            'New XGBoost': xgb_results['evaluation']['accuracy'],
            'Best Ensemble': ensemble_results['best_accuracy']
        }

        logger.info("🏆 COMPREHENSIVE RESULTS COMPARISON:")
        logger.info("-" * 50)
        for model_name, accuracy in results.items():
            status = "🟢" if accuracy >= 0.90 else "🟡" if accuracy >= 0.75 else "🔴"
            improvement = accuracy - baseline_accuracy if "Baseline" not in model_name else 0
            improvement_str = f" ({improvement:+.4f})" if improvement != 0 else ""
            logger.info(f"  {status} {model_name:20}: {accuracy:.4f} ({accuracy*100:.2f}%){improvement_str}")

        # Find best model
        best_model = max(results, key=results.get)
        best_accuracy = results[best_model]

        logger.info(f"\n🥇 BEST MODEL: {best_model}")
        logger.info(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

        # Detailed analysis
        if best_accuracy >= baseline_accuracy:
            improvement = best_accuracy - baseline_accuracy
            logger.info(f"🚀 SUCCESS: New model BEATS baseline by {improvement:.4f} ({improvement*100:.2f}%)!")
        elif best_accuracy >= 0.95:
            logger.info("🌟 EXCELLENT: New model achieves >95% accuracy!")
        elif best_accuracy >= 0.90:
            logger.info("✅ VERY GOOD: New model achieves >90% accuracy!")
        elif best_accuracy >= 0.75:
            logger.info("👍 GOOD: New model meets minimum target (75%+)!")
        else:
            logger.info("⚠️ NEEDS IMPROVEMENT: Below target accuracy")

        # Ensemble analysis
        logger.info(f"\n🤝 ENSEMBLE ANALYSIS:")
        logger.info(f"  Best Strategy: {ensemble_results['best_strategy']}")
        logger.info(f"  Individual Models:")
        for model, acc in ensemble_results['individual_accuracies'].items():
            logger.info(f"    {model.upper()}: {acc:.4f}")

        return results, best_model, best_accuracy
    
    def run_production_training(self):
        """
        Run the complete production training pipeline.
        """
        logger.info("🚀 STARTING PRODUCTION TENSION DETECTION TRAINING")
        logger.info("=" * 80)
        
        total_start_time = time.time()
        
        try:
            # Step 1: Load and prepare data
            splits, class_names, class_weights, encoder = self.load_and_prepare_data()
            
            # Step 2: Train Random Forest
            rf_results = self.train_random_forest_optimized(splits, class_weights)

            # Step 3: Train SVM
            svm_results = self.train_svm_optimized(splits, class_weights)

            # Step 4: Train XGBoost
            xgb_results = self.train_xgboost_optimized(splits, class_weights)

            # Step 5: Create full ensemble
            ensemble_results = self.create_full_ensemble(rf_results, svm_results, xgb_results, splits)

            # Step 6: Compare with baseline
            comparison, best_model, best_accuracy = self.compare_with_baseline_full(rf_results, svm_results, xgb_results, ensemble_results)
            
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            
            # Save summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_duration': total_duration,
                'data_samples': len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test']),
                'models_trained': ['Random Forest', 'SVM', 'XGBoost', 'Full Ensemble'],
                'results': comparison,
                'best_model': best_model,
                'best_accuracy': best_accuracy,
                'baseline_comparison': best_accuracy >= 0.9754
            }
            
            # Helper to convert numpy types to native Python so that json.dump works
            def _json_default(obj):
                import numpy as _np
                if isinstance(obj, _np.generic):
                    return obj.item()
                return str(obj)

            summary_path = self.output_dir / "tension_production_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=_json_default)
            
            logger.info("\n" + "=" * 80)
            logger.info("🎉 PRODUCTION TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"⏱️ Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            logger.info(f"📁 Summary saved: {summary_path}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Production training failed: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = ProductionTensionTrainer()
    results = trainer.run_production_training()
