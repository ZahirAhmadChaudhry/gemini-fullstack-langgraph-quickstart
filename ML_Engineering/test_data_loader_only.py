"""
Simple test script for just the data loader with new JSON format.
"""

import sys
import os
import logging

# Add the baseline_nlp directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'baseline_nlp'))

from baseline_nlp.utils.data_loader import DataLoader

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
        # Initialize data loader with data directory
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
            
            # Show sample data
            logger.info(f"Sample text: {sample_segment.get('text', '')[:100]}...")
            logger.info(f"Sample features: {features}")
            
            # Test feature validation
            temporal_context = features.get("temporal_context", "unknown")
            word_count = features.get("word_count", 0)
            noun_phrases = features.get("noun_phrases", [])
            
            logger.info(f"Temporal context: {temporal_context}")
            logger.info(f"Word count: {word_count}")
            logger.info(f"Number of noun phrases: {len(noun_phrases)}")
            
            return True
        else:
            logger.error("No segments extracted")
            return False
            
    except Exception as e:
        logger.error(f"Error testing data loader: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_engineering_basic():
    """Test basic feature engineering without ML dependencies."""
    logger.info("Testing basic feature engineering...")
    
    try:
        # Create sample segment
        sample_segment = {
            "id": "test_seg_001",
            "text": "Le changement climatique est un défi majeur pour notre société. Nous devons agir maintenant.",
            "features": {
                "temporal_context": "present",
                "discourse_markers": ["context"],
                "sentence_count": 2,
                "word_count": 14,
                "noun_phrases": ["changement climatique", "défi majeur", "notre société"]
            },
            "segment_id": "test_seg_001",
            "source_doc_id": "test_doc"
        }
        
        # Test basic feature extraction without spaCy
        from ml_pipeline.unsupervised_learning.feature_engineering import FeatureEngineering
        
        # This should work even without spaCy
        feature_eng = FeatureEngineering()
        enhanced_segment = feature_eng.extract_enhanced_features(sample_segment)
        
        logger.info("Feature engineering test completed")
        logger.info(f"Enhanced features keys: {list(enhanced_segment.get('features', {}).keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run basic tests."""
    logger.info("Starting Basic Tests...")
    
    tests = [
        ("DataLoader with new JSON format", test_data_loader_with_new_format),
        ("Basic Feature Engineering", test_feature_engineering_basic)
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
