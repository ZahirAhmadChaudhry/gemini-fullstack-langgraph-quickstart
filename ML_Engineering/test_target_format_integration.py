#!/usr/bin/env python3
"""
Test script for Target Format Integration with Enhanced Data Engineering Pipeline.

This script tests the integration of the target format generator with the ML pipeline
to ensure it can generate output matching the exact data.json structure.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_target_format_integration():
    """Test the complete target format integration."""
    logger.info("Testing Target Format Integration with Enhanced Data Engineering Pipeline...")
    
    try:
        # Import components
        from baseline_nlp.utils.data_loader import DataLoader
        from ml_pipeline.ml_integration import MLPipeline
        
        # Initialize data loader with new data from data engineering
        data_loader = DataLoader("data_from_Data_Engineering")
        
        # Load ML-ready data
        data = data_loader.load_file("_qHln3fOjOg_ml_ready.json")
        segments = data_loader.extract_segments(data)
        
        logger.info(f"Successfully loaded {len(segments)} segments from enhanced data engineering pipeline")
        
        # Initialize ML pipeline
        ml_pipeline = MLPipeline()
        
        # Check pipeline status
        status = ml_pipeline.get_pipeline_status()
        logger.info("Pipeline component status:")
        for component, info in status.items():
            logger.info(f"  {component}: {info}")
        
        # Test target format generation directly
        if ml_pipeline.target_format_generator:
            logger.info("Testing target format generation...")
            
            # Use first few segments for testing
            test_segments = segments[:3]
            
            target_format_data = ml_pipeline.generate_target_format(test_segments)
            
            if target_format_data.get("entries"):
                logger.info(f"Successfully generated {len(target_format_data['entries'])} target format entries")
                
                # Display sample entry
                sample_entry = target_format_data["entries"][0]
                logger.info("Sample target format entry:")
                for key, value in sample_entry.items():
                    logger.info(f"  {key}: {value}")
                
                # Validate structure
                required_columns = [
                    "Concepts de 2nd ordre",
                    "Items de 1er ordre reformulé", 
                    "Items de 1er ordre (intitulé d'origine)",
                    "Détails",
                    "Période",
                    "Thème",
                    "Code spé"
                ]
                
                missing_columns = []
                for column in required_columns:
                    if column not in sample_entry:
                        missing_columns.append(column)
                
                if missing_columns:
                    logger.error(f"Missing required columns: {missing_columns}")
                    return False
                else:
                    logger.info("✅ All required columns present in target format")
                
                # Validate data types and values
                validation_results = validate_target_format_entry(sample_entry)
                if validation_results["valid"]:
                    logger.info("✅ Target format entry validation passed")
                else:
                    logger.error(f"❌ Target format validation failed: {validation_results['errors']}")
                    return False
                
            else:
                logger.error("❌ No target format entries generated")
                return False
        else:
            logger.error("❌ Target format generator not available")
            return False
        
        # Test full pipeline with target format output
        logger.info("Testing full ML pipeline with target format output...")
        
        # Process segments through ML pipeline
        results = ml_pipeline.process_segments(
            test_segments,
            enable_topic_modeling=False,  # Skip for faster testing
            enable_semantic_search=False,
            enable_feature_engineering=True
        )
        
        # Save results with target format
        output_dir = "test_output"
        saved_files = ml_pipeline.save_results(
            results, 
            output_dir, 
            prefix="test_target_format_integration",
            generate_target_format=True
        )
        
        logger.info("Saved files:")
        for file_type, path in saved_files.items():
            logger.info(f"  {file_type}: {path}")
        
        # Verify target format file was created
        if "target_format" in saved_files:
            target_format_path = saved_files["target_format"]
            if os.path.exists(target_format_path):
                logger.info("✅ Target format file successfully created")
                
                # Load and validate the saved target format
                with open(target_format_path, 'r', encoding='utf-8') as f:
                    saved_target_data = json.load(f)
                
                if saved_target_data.get("entries"):
                    logger.info(f"✅ Saved target format contains {len(saved_target_data['entries'])} entries")
                else:
                    logger.error("❌ Saved target format file is empty")
                    return False
            else:
                logger.error("❌ Target format file was not created")
                return False
        else:
            logger.error("❌ Target format not included in saved files")
            return False
        
        logger.info("✅ Target Format Integration Test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Target Format Integration Test FAILED: {str(e)}", exc_info=True)
        return False


def validate_target_format_entry(entry):
    """Validate a target format entry against expected structure and values."""
    errors = []
    
    # Check Période values
    periode = entry.get("Période")
    if periode not in [2023.0, 2050.0, 2035.0]:
        errors.append(f"Invalid Période value: {periode} (expected 2023.0, 2050.0, or 2035.0)")
    
    # Check Thème values
    theme = entry.get("Thème")
    if theme not in ["Performance", "Légitimité"]:
        errors.append(f"Invalid Thème value: {theme} (expected 'Performance' or 'Légitimité')")
    
    # Check Concepts de 2nd ordre
    concept = entry.get("Concepts de 2nd ordre")
    expected_concepts = ["MODELES SOCIO-ECONOMIQUES", "MODELES ORGANISATIONNELS", "MODELES ENVIRONNEMENTAUX"]
    if concept not in expected_concepts:
        errors.append(f"Invalid Concepts de 2nd ordre: {concept} (expected one of {expected_concepts})")
    
    # Check that Détails is not empty
    details = entry.get("Détails", "")
    if not details.strip():
        errors.append("Détails field is empty")
    
    # Check that Code spé follows expected pattern
    code_spe = entry.get("Code spé", "")
    if not code_spe.startswith("10.tensions."):
        errors.append(f"Invalid Code spé format: {code_spe} (expected to start with '10.tensions.')")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def main():
    """Main entry point."""
    logger.info("Starting Target Format Integration Test...")
    
    success = test_target_format_integration()
    
    if success:
        logger.info("🎉 All tests passed! Target format integration is working correctly.")
        return 0
    else:
        logger.error("💥 Tests failed! Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
