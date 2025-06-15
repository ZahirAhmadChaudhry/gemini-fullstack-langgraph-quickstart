"""
Test script for the ML pipeline with new JSON format.

This script tests the enhanced ML pipeline with the new JSON input format
and validates all components work correctly.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the baseline_nlp directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'baseline_nlp'))

from baseline_nlp.utils.data_loader import DataLoader

# Try to import ML pipeline, but handle gracefully if dependencies are missing
try:
    from ml_pipeline.ml_integration import MLPipeline
    ML_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"ML Pipeline not fully available: {e}")
    ML_PIPELINE_AVAILABLE = False
    MLPipeline = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_loader_with_new_format():
    """Test the enhanced data loader with new JSON format."""
    logger.info("Testing DataLoader with new JSON format...")
    
    # Path to the new JSON file
    json_path = "preprocessed_data_by_Data_Engineer/_qHln3fOjOg_ml_ready.json"
    
    if not os.path.exists(json_path):
        logger.error(f"Test JSON file not found: {json_path}")
        return False
    
    try:
        # Initialize data loader
        data_loader = DataLoader("preprocessed_data_by_Data_Engineer")
        
        # Load and extract segments
        data = data_loader.load_file("_qHln3fOjOg_ml_ready.json")
        segments = data_loader.extract_segments(data)
        
        logger.info(f"Successfully loaded {len(segments)} segments")
        
        # Validate segment structure
        if segments:
            sample_segment = segments[0]
            logger.info(f"Sample segment keys: {list(sample_segment.keys())}")
            
            # Check for required fields
            required_fields = ["text", "segment_id", "features"]
            for field in required_fields:
                if field not in sample_segment:
                    logger.warning(f"Missing required field: {field}")
                else:
                    logger.info(f"✓ Found required field: {field}")
            
            # Check features structure
            features = sample_segment.get("features", {})
            logger.info(f"Features keys: {list(features.keys())}")
            
            return True
        else:
            logger.error("No segments extracted")
            return False
            
    except Exception as e:
        logger.error(f"Error testing data loader: {e}")
        return False

def test_ml_pipeline():
    """Test the complete ML pipeline."""
    logger.info("Testing ML Pipeline...")

    if not ML_PIPELINE_AVAILABLE:
        logger.warning("ML Pipeline not available due to missing dependencies")
        return False

    try:
        # Create sample data that matches the new format
        sample_segments = [
            {
                "id": "test_seg_001",
                "text": "Le changement climatique est un défi majeur pour notre société. Nous devons agir maintenant pour réduire les émissions.",
                "features": {
                    "temporal_context": "present",
                    "discourse_markers": ["context"],
                    "sentence_count": 2,
                    "word_count": 18,
                    "noun_phrases": ["changement climatique", "défi majeur", "notre société", "les émissions"]
                },
                "metadata": {
                    "source": "test_data",
                    "segment_lines": 2,
                    "position": {"start": 0, "end": 2}
                },
                "segment_id": "test_seg_001",
                "source_doc_id": "test_doc"
            },
            {
                "id": "test_seg_002", 
                "text": "L'avenir de l'emploi dépend de notre capacité à nous adapter aux nouvelles technologies et à l'automatisation.",
                "features": {
                    "temporal_context": "future",
                    "discourse_markers": ["dependency"],
                    "sentence_count": 1,
                    "word_count": 16,
                    "noun_phrases": ["l'avenir", "l'emploi", "notre capacité", "nouvelles technologies", "l'automatisation"]
                },
                "metadata": {
                    "source": "test_data",
                    "segment_lines": 1,
                    "position": {"start": 3, "end": 3}
                },
                "segment_id": "test_seg_002",
                "source_doc_id": "test_doc"
            }
        ]
        
        # Initialize ML pipeline
        logger.info("Initializing ML pipeline...")
        ml_pipeline = MLPipeline()
        
        # Check pipeline status
        status = ml_pipeline.get_pipeline_status()
        logger.info(f"Pipeline status: {status}")
        
        # Process segments (with limited features due to missing dependencies)
        logger.info("Processing segments...")
        
        # Test feature engineering only (doesn't require external models)
        results = ml_pipeline.process_segments(
            sample_segments,
            enable_topic_modeling=False,  # Requires BERTopic
            enable_semantic_search=False,  # Requires FAISS
            enable_feature_engineering=True
        )
        
        logger.info(f"Processing results keys: {list(results.keys())}")
        logger.info(f"Processed {results.get('input_segments', 0)} segments")
        
        # Test dataset splitting
        logger.info("Testing dataset splitting...")
        split_data = ml_pipeline.split_dataset(sample_segments, stratify_by="temporal_context")

        for split_name, split_segments in split_data.items():
            logger.info(f"{split_name}: {len(split_segments)} segments")

        # Validate split
        validation_report = ml_pipeline.data_splitter.validate_split(split_data)
        logger.info(f"Split validation: {validation_report['data_leakage_check']['has_leakage']}")

        # Test Excel export
        logger.info("Testing Excel export...")
        output_dir = "test_output"
        saved_files = ml_pipeline.save_results(results, output_dir, "test_ml_pipeline")

        if "excel_report" in saved_files:
            logger.info(f"✅ Excel report generated: {saved_files['excel_report']}")
        else:
            logger.warning("⚠️ Excel report not generated (openpyxl may not be available)")

        logger.info(f"Saved {len(saved_files)} files: {list(saved_files.keys())}")

        return True
        
    except Exception as e:
        logger.error(f"Error testing ML pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data():
    """Test with real data from the preprocessed file."""
    logger.info("Testing with real data...")

    if not ML_PIPELINE_AVAILABLE:
        logger.warning("ML Pipeline not available due to missing dependencies")
        return False

    json_path = "preprocessed_data_by_Data_Engineer/_qHln3fOjOg_ml_ready.json"

    if not os.path.exists(json_path):
        logger.warning(f"Real data file not found: {json_path}")
        return False

    try:
        # Load real data
        data_loader = DataLoader("preprocessed_data_by_Data_Engineer")
        data = data_loader.load_file("_qHln3fOjOg_ml_ready.json")
        segments = data_loader.extract_segments(data)
        
        if not segments:
            logger.error("No segments loaded from real data")
            return False
        
        logger.info(f"Loaded {len(segments)} real segments")
        
        # Test with first few segments to avoid long processing
        test_segments = segments[:5]
        
        # Initialize ML pipeline
        ml_pipeline = MLPipeline()
        
        # Process with feature engineering only
        results = ml_pipeline.process_segments(
            test_segments,
            enable_topic_modeling=False,
            enable_semantic_search=False,
            enable_feature_engineering=True
        )
        
        logger.info("Real data processing completed successfully")

        # Show sample enhanced features
        if results.get("processed_segments"):
            sample = results["processed_segments"][0]
            features = sample.get("features", {})
            logger.info(f"Enhanced features sample: {list(features.keys())}")

        # Test Excel export with real data
        logger.info("Testing Excel export with real data...")
        output_dir = "real_data_output"
        saved_files = ml_pipeline.save_results(results, output_dir, "real_data_analysis")

        if "excel_report" in saved_files:
            logger.info(f"✅ Real data Excel report generated: {saved_files['excel_report']}")
        else:
            logger.warning("⚠️ Excel report not generated (openpyxl may not be available)")

        logger.info(f"Real data analysis saved to {len(saved_files)} files")

        return True
        
    except Exception as e:
        logger.error(f"Error testing with real data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("Starting ML Pipeline Tests...")
    
    tests = [
        ("DataLoader with new JSON format", test_data_loader_with_new_format),
        ("ML Pipeline with sample data", test_ml_pipeline),
        ("ML Pipeline with real data", test_with_real_data)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
            logger.info(f"Test {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            logger.error(f"Test {test_name} failed with error: {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    passed_tests = sum(1 for result in results.values() if result == "PASSED")
    total_tests = len(results)
    
    logger.info(f"\nPassed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
