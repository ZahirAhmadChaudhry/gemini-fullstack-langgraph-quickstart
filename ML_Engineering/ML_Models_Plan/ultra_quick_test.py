"""
Ultra Quick Model Testing - No Data Loading
==========================================

Test models using synthetic data or cached small datasets to avoid 
the time-consuming data loading process.

Author: ML Engineering Team
Date: 2025-06-12
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.tension_detection.random_forest import TensionRandomForestModel
from models.tension_detection.svm_model import TensionSVMModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_tension_data(n_samples=100, n_features=50):
    """
    Create synthetic data that mimics the real tension detection dataset structure.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, feature_names, class_names)
    """
    print(f"🔧 Creating synthetic tension data: {n_samples} samples, {n_features} features")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate feature names (similar to real data)
    feature_names = [
        'word_count', 'sentence_count', 'temporal_period', 'conceptual_complexity',
        'tension_strength_score', 'discourse_marker_count', 'pos_noun_ratio',
        'pos_verb_ratio', 'pos_adj_ratio', 'noun_phrase_count',
        'sustainability_score', 'temporal_confidence', 'lexical_diversity'
    ]
    
    # Add more synthetic features to reach n_features
    while len(feature_names) < n_features:
        feature_names.append(f'synthetic_feature_{len(feature_names)}')
    
    feature_names = feature_names[:n_features]
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Make features more realistic
    X[:, 0] = np.random.randint(10, 200, n_samples)  # word_count
    X[:, 1] = np.random.randint(1, 10, n_samples)    # sentence_count
    X[:, 2] = np.random.choice([2025, 2030, 2035, 2040], n_samples)  # temporal_period
    X[:, 3] = np.random.uniform(0, 1, n_samples)     # conceptual_complexity
    X[:, 4] = np.random.uniform(0, 1, n_samples)     # tension_strength_score
    
    # Generate labels (5 tension types like real data)
    class_names = ['dévelpmt', 'NV', 'temps', 'richesse', 'travail']
    y = np.random.choice(range(len(class_names)), n_samples, 
                        p=[0.7, 0.15, 0.05, 0.05, 0.05])  # Imbalanced like real data
    
    # Split into train/val
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"✅ Generated {len(X_train)} train, {len(X_val)} val samples")
    print(f"✅ Class distribution: {dict(zip(class_names, np.bincount(y, minlength=len(class_names))))}")
    
    return X_train, y_train, X_val, y_val, feature_names, class_names

def create_synthetic_text_data(n_samples=50):
    """
    Create synthetic text data for thematic classification testing.
    
    Args:
        n_samples: Number of text samples to generate
        
    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels, class_names)
    """
    print(f"🔧 Creating synthetic text data: {n_samples} samples")
    
    # Sample French texts for each theme
    performance_texts = [
        "L'efficacité de cette solution dépend de la performance économique.",
        "Les résultats montrent une amélioration significative de la productivité.",
        "Cette approche optimise les ressources et maximise les bénéfices.",
        "L'objectif principal est d'atteindre une performance durable.",
        "Les indicateurs de performance révèlent des tendances positives."
    ]
    
    legitimacy_texts = [
        "Cette décision doit être acceptée par toutes les parties prenantes.",
        "La légitimité de cette approche repose sur le consensus social.",
        "Il est important d'obtenir l'adhésion de la communauté locale.",
        "Cette solution respecte les valeurs et principes établis.",
        "L'acceptabilité sociale est un critère fondamental."
    ]
    
    # Generate synthetic texts
    texts = []
    labels = []
    class_names = ['Performance', 'Légitimité']
    
    for i in range(n_samples):
        if i % 2 == 0:  # Performance (50%)
            base_text = np.random.choice(performance_texts)
            texts.append(f"{base_text} Contexte {i}: développement et innovation.")
            labels.append(0)
        else:  # Légitimité (50%)
            base_text = np.random.choice(legitimacy_texts)
            texts.append(f"{base_text} Contexte {i}: validation et reconnaissance.")
            labels.append(1)
    
    # Split into train/val
    split_idx = int(0.8 * n_samples)
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    print(f"✅ Generated {len(train_texts)} train, {len(val_texts)} val texts")
    
    return train_texts, train_labels, val_texts, val_labels, class_names

def ultra_quick_svm_test():
    """
    Ultra quick SVM test with synthetic data.
    """
    print("🧪 ULTRA QUICK SVM TEST")
    print("=" * 30)
    
    try:
        # 1. Generate synthetic data (instant)
        X_train, y_train, X_val, y_val, feature_names, class_names = create_synthetic_tension_data(100, 30)
        
        # 2. Train SVM model
        print("\n2️⃣ Training SVM model...")
        svm_model = TensionSVMModel()
        
        results = svm_model.train(
            X_train, y_train,
            X_val, y_val,
            optimize=False  # No optimization for speed
        )
        
        print(f"✅ SVM training completed!")
        print(f"  Training accuracy: {results['train_accuracy']:.4f}")
        if 'val_accuracy' in results:
            print(f"  Validation accuracy: {results['val_accuracy']:.4f}")
        
        # 3. Test predictions
        predictions = svm_model.predict(X_val)
        print(f"✅ Generated {len(predictions)} predictions")
        
        return results
        
    except Exception as e:
        print(f"❌ SVM test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def ultra_quick_ensemble_test():
    """
    Ultra quick Ensemble test with synthetic data.
    """
    print("\n🧪 ULTRA QUICK ENSEMBLE TEST")
    print("=" * 30)
    
    try:
        from models.tension_detection.ensemble_model import TensionEnsembleModel
        
        # 1. Generate synthetic data
        X_train, y_train, X_val, y_val, feature_names, class_names = create_synthetic_tension_data(60, 25)
        
        # 2. Train ensemble model
        print("\n2️⃣ Training Ensemble model...")
        ensemble_model = TensionEnsembleModel()
        
        results = ensemble_model.train(
            X_train, y_train,
            X_val, y_val,
            optimize=True,
            n_trials=3  # Minimal optimization
        )
        
        print(f"✅ Ensemble training completed!")
        print(f"  Training accuracy: {results['train_accuracy']:.4f}")
        if 'val_accuracy' in results:
            print(f"  Validation accuracy: {results['val_accuracy']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ensemble test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def ultra_quick_thematic_test():
    """
    Ultra quick thematic models test with synthetic text data.
    """
    print("\n🧪 ULTRA QUICK THEMATIC TEST")
    print("=" * 30)
    
    try:
        from models.thematic_classification.logistic_regression import ThematicLogisticRegressionModel
        from models.thematic_classification.naive_bayes import ThematicNaiveBayesModel
        
        # 1. Generate synthetic text data
        train_texts, train_labels, val_texts, val_labels, class_names = create_synthetic_text_data(40)
        
        # 2. Test Logistic Regression
        print("\n2️⃣ Testing Logistic Regression...")
        lr_model = ThematicLogisticRegressionModel()
        lr_results = lr_model.train(
            train_texts, train_labels,
            val_texts, val_labels,
            optimize=False
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
            optimize=False
        )
        
        print(f"✅ Naive Bayes completed!")
        print(f"  Training accuracy: {nb_results['train_accuracy']:.4f}")
        if 'val_accuracy' in nb_results:
            print(f"  Validation accuracy: {nb_results['val_accuracy']:.4f}")
        
        return {
            'logistic_regression': lr_results,
            'naive_bayes': nb_results
        }
        
    except Exception as e:
        print(f"❌ Thematic test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_real_data_subset():
    """
    Save a small subset of real data for future quick testing.
    Run this once to create cached data.
    """
    print("💾 SAVING REAL DATA SUBSET FOR FUTURE QUICK TESTS...")
    
    try:
        from data_preparation import DataPreparationPipeline
        
        # Load real data
        pipeline = DataPreparationPipeline('../data_from_Data_Engineering')
        target_df, ml_df = pipeline.load_all_data()
        
        # Prepare datasets
        tension_dataset = pipeline.prepare_tension_dataset(target_df, ml_df)
        thematic_dataset = pipeline.prepare_thematic_dataset(target_df, ml_df)
        
        # Create splits
        tension_splits = pipeline.create_train_test_splits(tension_dataset, 'tension_label')
        thematic_splits = pipeline.create_train_test_splits(thematic_dataset, 'theme_label')
        
        # Save small subsets
        subset_data = {
            'tension': {
                'X_train': tension_splits['X_train'][:100],
                'y_train': tension_splits['y_train'][:100],
                'X_val': tension_splits['X_val'][:20],
                'y_val': tension_splits['y_val'][:20],
                'feature_columns': tension_splits['feature_columns']
            },
            'thematic': {
                'train_texts': thematic_dataset.iloc[thematic_splits['train_indices'][:50]]['text'].tolist(),
                'train_labels': thematic_splits['y_train'][:50].tolist(),
                'val_texts': thematic_dataset.iloc[thematic_splits['val_indices'][:10]]['text'].tolist(),
                'val_labels': thematic_splits['y_val'][:10].tolist()
            },
            'encoders': {
                'tension': pipeline.tension_encoder,
                'thematic': pipeline.theme_encoder
            }
        }
        
        # Save to pickle file
        cache_file = Path('cached_test_data.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(subset_data, f)
        
        print(f"✅ Cached test data saved to {cache_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save cached data: {str(e)}")
        return False

def main():
    """
    Run ultra quick tests with synthetic data.
    """
    print("🚀 STARTING ULTRA QUICK MODEL TESTS (SYNTHETIC DATA)")
    print("=" * 60)
    
    # Test 1: SVM model
    svm_results = ultra_quick_svm_test()
    
    # Test 2: Ensemble model  
    ensemble_results = ultra_quick_ensemble_test()
    
    # Test 3: Thematic models
    thematic_results = ultra_quick_thematic_test()
    
    print("\n" + "=" * 60)
    print("🏁 ALL ULTRA QUICK TESTS COMPLETED!")
    
    if svm_results and ensemble_results and thematic_results:
        print("✅ All models work correctly with synthetic data")
        print("🚀 Models are ready for real data training!")
        print("\n💡 Next steps:")
        print("  1. Run full training with real data")
        print("  2. Compare performance with Random Forest baseline")
        print("  3. Select best models for production")
    else:
        print("⚠️ Some tests failed - check model implementations")
    
    return {
        'svm': svm_results,
        'ensemble': ensemble_results,
        'thematic': thematic_results
    }

if __name__ == "__main__":
    results = main()
