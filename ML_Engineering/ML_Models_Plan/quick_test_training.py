"""
Quick Test Training Script - Small Data Subset
==============================================

Test training new models on a small subset of data for rapid iteration.
Uses the exact same data preparation as the successful Random Forest model.

Author: ML Engineering Team
Date: 2025-06-12
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_preparation import DataPreparationPipeline
from models.tension_detection.random_forest import TensionRandomForestModel
from models.tension_detection.svm_model import TensionSVMModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_test_svm_model():
    """
    Quick test of SVM model on small data subset.
    """
    print("🧪 QUICK SVM MODEL TEST")
    print("=" * 40)
    
    try:
        # 1. Load data (reuse existing pipeline)
        print("\n1️⃣ Loading small data subset...")
        pipeline = DataPreparationPipeline('../data_from_Data_Engineering')
        target_df, ml_df = pipeline.load_all_data()
        
        # Prepare tension dataset
        tension_dataset = pipeline.prepare_tension_dataset(target_df, ml_df)
        print(f"✅ Full dataset: {len(tension_dataset)} samples")
        
        # Create splits
        splits = pipeline.create_train_test_splits(tension_dataset, 'tension_label')
        
        # 2. Use small subset for quick testing
        SUBSET_SIZE = 100  # Small subset for quick testing
        
        X_train_small = splits['X_train'][:SUBSET_SIZE]
        y_train_small = splits['y_train'][:SUBSET_SIZE]
        X_val_small = splits['X_val'][:20]  # Even smaller validation set
        y_val_small = splits['y_val'][:20]
        
        print(f"✅ Using subset: {len(X_train_small)} train, {len(X_val_small)} val samples")
        
        # 3. Train SVM model (no optimization for speed)
        print("\n2️⃣ Training SVM model...")
        svm_model = TensionSVMModel()
        
        # Quick training without optimization
        results = svm_model.train(
            X_train_small, y_train_small,
            X_val_small, y_val_small,
            optimize=False  # Skip optimization for speed
        )
        
        print(f"✅ SVM training completed!")
        print(f"  Training accuracy: {results['train_accuracy']:.4f}")
        if 'val_accuracy' in results:
            print(f"  Validation accuracy: {results['val_accuracy']:.4f}")
        
        # 4. Test predictions
        print("\n3️⃣ Testing predictions...")
        predictions = svm_model.predict(X_val_small)
        print(f"✅ Generated {len(predictions)} predictions")
        
        # 5. Compare with Random Forest on same data
        print("\n4️⃣ Comparing with Random Forest...")
        rf_model = TensionRandomForestModel()
        rf_results = rf_model.train(
            X_train_small, y_train_small,
            X_val_small, y_val_small,
            optimize=False
        )
        
        print(f"✅ Random Forest comparison:")
        print(f"  RF Training accuracy: {rf_results['train_accuracy']:.4f}")
        if 'val_accuracy' in rf_results:
            print(f"  RF Validation accuracy: {rf_results['val_accuracy']:.4f}")
        
        print("\n🎉 QUICK TEST COMPLETED SUCCESSFULLY!")
        
        return {
            'svm_results': results,
            'rf_results': rf_results,
            'subset_size': SUBSET_SIZE
        }
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def quick_test_ensemble_model():
    """
    Quick test of Ensemble model on small data subset.
    """
    print("\n🧪 QUICK ENSEMBLE MODEL TEST")
    print("=" * 40)
    
    try:
        # Import ensemble model
        from models.tension_detection.ensemble_model import TensionEnsembleModel
        
        # 1. Load data (reuse existing pipeline)
        print("\n1️⃣ Loading small data subset...")
        pipeline = DataPreparationPipeline('../data_from_Data_Engineering')
        target_df, ml_df = pipeline.load_all_data()
        
        # Prepare tension dataset
        tension_dataset = pipeline.prepare_tension_dataset(target_df, ml_df)
        
        # Create splits
        splits = pipeline.create_train_test_splits(tension_dataset, 'tension_label')
        
        # 2. Use small subset for quick testing
        SUBSET_SIZE = 50  # Even smaller for ensemble (trains 3 models)
        
        X_train_small = splits['X_train'][:SUBSET_SIZE]
        y_train_small = splits['y_train'][:SUBSET_SIZE]
        X_val_small = splits['X_val'][:15]
        y_val_small = splits['y_val'][:15]
        
        print(f"✅ Using subset: {len(X_train_small)} train, {len(X_val_small)} val samples")
        
        # 3. Train ensemble model (minimal optimization)
        print("\n2️⃣ Training Ensemble model...")
        ensemble_model = TensionEnsembleModel()
        
        # Quick training with minimal optimization
        results = ensemble_model.train(
            X_train_small, y_train_small,
            X_val_small, y_val_small,
            optimize=True,
            n_trials=5  # Very few trials for speed
        )
        
        print(f"✅ Ensemble training completed!")
        print(f"  Training accuracy: {results['train_accuracy']:.4f}")
        if 'val_accuracy' in results:
            print(f"  Validation accuracy: {results['val_accuracy']:.4f}")
        
        # 4. Test predictions
        print("\n3️⃣ Testing predictions...")
        predictions = ensemble_model.predict(X_val_small)
        print(f"✅ Generated {len(predictions)} predictions")
        
        print("\n🎉 ENSEMBLE QUICK TEST COMPLETED!")
        
        return results
        
    except Exception as e:
        print(f"❌ Ensemble quick test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def quick_test_thematic_models():
    """
    Quick test of thematic classification models on small data subset.
    """
    print("\n🧪 QUICK THEMATIC MODELS TEST")
    print("=" * 40)
    
    try:
        # Import thematic models
        from models.thematic_classification.logistic_regression import ThematicLogisticRegressionModel
        from models.thematic_classification.naive_bayes import ThematicNaiveBayesModel
        
        # 1. Load data
        print("\n1️⃣ Loading thematic data subset...")
        pipeline = DataPreparationPipeline('../data_from_Data_Engineering')
        target_df, ml_df = pipeline.load_all_data()
        
        # Prepare thematic dataset
        thematic_dataset = pipeline.prepare_thematic_dataset(target_df, ml_df)
        print(f"✅ Full thematic dataset: {len(thematic_dataset)} samples")
        
        # Create splits
        thematic_splits = pipeline.create_train_test_splits(thematic_dataset, 'theme_label')
        
        # Get text data for thematic models
        train_indices = thematic_splits['train_indices'][:50]  # Small subset
        val_indices = thematic_splits['val_indices'][:15]
        
        train_texts = thematic_dataset.iloc[train_indices]['text'].tolist()
        val_texts = thematic_dataset.iloc[val_indices]['text'].tolist()
        train_labels = thematic_splits['y_train'][:50].tolist()
        val_labels = thematic_splits['y_val'][:15].tolist()
        
        print(f"✅ Using subset: {len(train_texts)} train, {len(val_texts)} val texts")
        
        # 2. Test Logistic Regression
        print("\n2️⃣ Testing Logistic Regression...")
        lr_model = ThematicLogisticRegressionModel()
        lr_results = lr_model.train(
            train_texts, train_labels,
            val_texts, val_labels,
            optimize=False  # Skip optimization for speed
        )
        
        print(f"✅ Logistic Regression completed!")
        print(f"  Training accuracy: {lr_results['train_accuracy']:.4f}")
        if 'val_accuracy' in lr_results:
            print(f"  Validation accuracy: {lr_results['val_accuracy']:.4f}")
        
        # 3. Test Naive Bayes
        print("\n3️⃣ Testing Naive Bayes...")
        nb_model = ThematicNaiveBayesModel()
        nb_results = nb_model.train(
            train_texts, train_labels,
            val_texts, val_labels,
            optimize=False  # Skip optimization for speed
        )
        
        print(f"✅ Naive Bayes completed!")
        print(f"  Training accuracy: {nb_results['train_accuracy']:.4f}")
        if 'val_accuracy' in nb_results:
            print(f"  Validation accuracy: {nb_results['val_accuracy']:.4f}")
        
        print("\n🎉 THEMATIC QUICK TESTS COMPLETED!")
        
        return {
            'logistic_regression': lr_results,
            'naive_bayes': nb_results
        }
        
    except Exception as e:
        print(f"❌ Thematic quick test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Run all quick tests.
    """
    print("🚀 STARTING QUICK MODEL TESTS")
    print("=" * 50)
    
    # Test 1: SVM model
    svm_results = quick_test_svm_model()
    
    # Test 2: Ensemble model
    ensemble_results = quick_test_ensemble_model()
    
    # Test 3: Thematic models
    thematic_results = quick_test_thematic_models()
    
    print("\n" + "=" * 50)
    print("🏁 ALL QUICK TESTS COMPLETED!")
    
    if svm_results and ensemble_results and thematic_results:
        print("✅ All models tested successfully on small data subsets")
        print("🚀 Ready to proceed with full-scale training!")
    else:
        print("⚠️ Some tests failed - check logs for details")
    
    return {
        'svm': svm_results,
        'ensemble': ensemble_results,
        'thematic': thematic_results
    }

if __name__ == "__main__":
    results = main()
