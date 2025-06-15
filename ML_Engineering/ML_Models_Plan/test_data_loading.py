"""
Test Data Loading and Preparation
=================================

Simple test script to verify data loading and preparation works correctly
before running full model training.

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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from data_preparation import DataPreparationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test basic data loading functionality."""
    logger.info("🧪 Testing Data Loading...")
    
    # Initialize pipeline
    data_dir = Path(__file__).parent.parent / 'data_from_Data_Engineering'
    pipeline = DataPreparationPipeline(str(data_dir))
    
    # Test 1: Check if directories exist
    logger.info(f"Data directory: {pipeline.data_dir}")
    logger.info(f"Target format directory: {pipeline.target_format_dir}")
    logger.info(f"ML ready directory: {pipeline.ml_ready_dir}")
    
    logger.info(f"Target format dir exists: {pipeline.target_format_dir.exists()}")
    logger.info(f"ML ready dir exists: {pipeline.ml_ready_dir.exists()}")
    
    # Test 2: List available files
    if pipeline.target_format_dir.exists():
        target_files = list(pipeline.target_format_dir.glob("*.json"))
        logger.info(f"Target format files found: {len(target_files)}")
        for f in target_files[:3]:  # Show first 3
            logger.info(f"  - {f.name}")
    
    if pipeline.ml_ready_dir.exists():
        ml_files = list(pipeline.ml_ready_dir.glob("*.json"))
        logger.info(f"ML ready files found: {len(ml_files)}")
        for f in ml_files[:3]:  # Show first 3
            logger.info(f"  - {f.name}")
    
    # Test 3: Load one file to check structure
    if pipeline.target_format_dir.exists():
        target_files = list(pipeline.target_format_dir.glob("*.json"))
        if target_files:
            logger.info(f"\n📄 Examining first target file: {target_files[0].name}")
            with open(target_files[0], 'r', encoding='utf-8') as f:
                sample_target = json.load(f)
                logger.info(f"Target file structure keys: {list(sample_target.keys())}")
                if 'entries' in sample_target:
                    logger.info(f"Number of entries: {len(sample_target['entries'])}")
                    if sample_target['entries']:
                        logger.info(f"First entry keys: {list(sample_target['entries'][0].keys())}")
                        logger.info(f"Sample entry: {sample_target['entries'][0]}")
    
    if pipeline.ml_ready_dir.exists():
        ml_files = list(pipeline.ml_ready_dir.glob("*.json"))
        if ml_files:
            logger.info(f"\n📄 Examining first ML file: {ml_files[0].name}")
            with open(ml_files[0], 'r', encoding='utf-8') as f:
                sample_ml = json.load(f)
                logger.info(f"ML file structure keys: {list(sample_ml.keys())}")
                if 'segments' in sample_ml:
                    logger.info(f"Number of segments: {len(sample_ml['segments'])}")
                    if sample_ml['segments']:
                        logger.info(f"First segment keys: {list(sample_ml['segments'][0].keys())}")
                        logger.info(f"Sample segment text: {sample_ml['segments'][0].get('text', 'N/A')[:100]}...")

def test_data_preparation():
    """Test the complete data preparation pipeline."""
    logger.info("\n🔧 Testing Data Preparation Pipeline...")
    
    try:
        # Initialize pipeline
        data_dir = Path(__file__).parent.parent / 'data_from_Data_Engineering'
        pipeline = DataPreparationPipeline(str(data_dir))
        
        # Test loading all data
        logger.info("Loading all data...")
        target_df, ml_df = pipeline.load_all_data()
        
        logger.info(f"✅ Loaded {len(target_df)} target entries")
        logger.info(f"✅ Loaded {len(ml_df)} ML segments")
        
        if len(target_df) > 0:
            logger.info(f"Target columns: {list(target_df.columns)}")
            logger.info(f"Sample target entry:")
            logger.info(target_df.iloc[0].to_dict())
        
        if len(ml_df) > 0:
            logger.info(f"ML columns: {list(ml_df.columns)}")
            logger.info(f"Sample ML segment:")
            logger.info(f"  ID: {ml_df.iloc[0].get('id', 'N/A')}")
            logger.info(f"  Text: {ml_df.iloc[0].get('text', 'N/A')[:100]}...")
            logger.info(f"  Features keys: {list(ml_df.iloc[0].get('features', {}).keys())}")
        
        # Test tension dataset preparation
        if len(target_df) > 0 and len(ml_df) > 0:
            logger.info("\n🎯 Testing tension dataset preparation...")
            tension_dataset = pipeline.prepare_tension_dataset(target_df, ml_df)
            logger.info(f"✅ Prepared tension dataset with {len(tension_dataset)} samples")
            
            if len(tension_dataset) > 0:
                logger.info(f"Tension dataset columns: {list(tension_dataset.columns)}")
                if 'tension_label' in tension_dataset.columns:
                    tension_counts = tension_dataset['tension_label'].value_counts()
                    logger.info(f"Tension label distribution: {tension_counts.to_dict()}")
                else:
                    logger.warning("No 'tension_label' column found in tension dataset")
            
            # Test thematic dataset preparation
            logger.info("\n🎨 Testing thematic dataset preparation...")
            thematic_dataset = pipeline.prepare_thematic_dataset(target_df, ml_df)
            logger.info(f"✅ Prepared thematic dataset with {len(thematic_dataset)} samples")
            
            if len(thematic_dataset) > 0:
                logger.info(f"Thematic dataset columns: {list(thematic_dataset.columns)}")
                if 'theme_label' in thematic_dataset.columns:
                    theme_counts = thematic_dataset['theme_label'].value_counts()
                    logger.info(f"Theme label distribution: {theme_counts.to_dict()}")
                else:
                    logger.warning("No 'theme_label' column found in thematic dataset")
            
            # Test train/test splits if we have data
            if len(tension_dataset) > 0 and 'tension_label' in tension_dataset.columns:
                logger.info("\n📊 Testing train/test splits...")
                try:
                    splits = pipeline.create_train_test_splits(tension_dataset, 'tension_label')
                    logger.info(f"✅ Created splits successfully")
                    logger.info(f"  Train samples: {len(splits['X_train'])}")
                    logger.info(f"  Val samples: {len(splits['X_val'])}")
                    logger.info(f"  Test samples: {len(splits['X_test'])}")
                    logger.info(f"  Feature count: {len(splits['feature_columns'])}")
                except Exception as e:
                    logger.error(f"❌ Split creation failed: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_small_model_training():
    """Test training a simple model on a small subset of data."""
    logger.info("\n🤖 Testing Small Model Training...")
    
    try:
        # Import model
        from models.tension_detection.random_forest import TensionRandomForestModel
        
        # Initialize pipeline and prepare data
        data_dir = Path(__file__).parent.parent / 'data_from_Data_Engineering'
        pipeline = DataPreparationPipeline(str(data_dir))
        target_df, ml_df = pipeline.load_all_data()
        
        if len(target_df) == 0 or len(ml_df) == 0:
            logger.error("❌ No data available for training test")
            return False
        
        # Prepare tension dataset
        tension_dataset = pipeline.prepare_tension_dataset(target_df, ml_df)
        
        if len(tension_dataset) == 0:
            logger.error("❌ No tension dataset prepared")
            return False
        
        if 'tension_label' not in tension_dataset.columns:
            logger.error("❌ No tension_label column found")
            return False
        
        # Create splits
        splits = pipeline.create_train_test_splits(tension_dataset, 'tension_label')
        
        # Get small subset for quick test (max 50 samples)
        max_samples = min(50, len(splits['X_train']))
        X_train_small = splits['X_train'][:max_samples]
        y_train_small = splits['y_train'][:max_samples]
        X_val_small = splits['X_val'][:min(20, len(splits['X_val']))]
        y_val_small = splits['y_val'][:min(20, len(splits['y_val']))]
        
        logger.info(f"Training on {len(X_train_small)} samples, validating on {len(X_val_small)} samples")
        
        # Initialize and train model (no optimization for speed)
        model = TensionRandomForestModel()
        results = model.train(
            X_train_small, y_train_small, 
            X_val_small, y_val_small,
            optimize=False  # Skip optimization for quick test
        )
        
        logger.info(f"✅ Model training completed!")
        logger.info(f"  Training accuracy: {results['train_accuracy']:.4f}")
        if 'val_accuracy' in results:
            logger.info(f"  Validation accuracy: {results['val_accuracy']:.4f}")
        
        # Test prediction
        predictions = model.predict(X_val_small)
        logger.info(f"✅ Predictions generated: {len(predictions)} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Small model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Data Loading and Preparation Tests")
    logger.info("=" * 60)
    
    # Test 1: Basic data loading
    test_data_loading()
    
    # Test 2: Data preparation pipeline
    success = test_data_preparation()
    
    if success:
        # Test 3: Small model training
        test_small_model_training()
    
    logger.info("\n" + "=" * 60)
    logger.info("🏁 Tests completed!")

if __name__ == "__main__":
    main()
