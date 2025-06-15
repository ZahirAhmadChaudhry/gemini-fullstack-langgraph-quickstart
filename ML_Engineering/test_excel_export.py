"""
Simple test for Excel export functionality.
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

def test_excel_export():
    """Test Excel export with real data."""
    logger.info("Testing Excel export functionality...")
    
    try:
        # Load real data
        data_loader = DataLoader("preprocessed_data_by_Data_Engineer")
        data = data_loader.load_file("_qHln3fOjOg_ml_ready.json")
        segments = data_loader.extract_segments(data)
        
        logger.info(f"Loaded {len(segments)} segments")
        
        # Create sample results structure
        results = {
            "input_segments": len(segments),
            "processed_segments": segments,
            "feature_engineering_results": {
                "enhanced_segments": len(segments),
                "features_added": True
            },
            "evaluation_results": {
                "summary": {
                    "total_segments": len(segments),
                    "data_quality_score": 0.95,
                    "overall_quality_score": 0.88
                },
                "detailed_metrics": {
                    "feature_quality": {
                        "total_segments": len(segments),
                        "data_quality": {
                            "feature_completeness_ratio": 0.95
                        }
                    }
                }
            },
            "processing_metadata": {
                "timestamp": "2025-01-11T17:30:00",
                "config": {
                    "feature_engineering": {"spacy_model": "fr_core_news_lg"}
                }
            }
        }
        
        # Test Excel export
        from ml_pipeline.utils.excel_exporter import ExcelExporter
        
        exporter = ExcelExporter()
        output_path = "test_excel_export.xlsx"
        
        success = exporter.export_ml_results(results, output_path, include_raw_data=True)
        
        if success:
            logger.info(f"✅ Excel export successful: {output_path}")
            
            # Check if file exists
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Excel file created successfully: {file_size} bytes")
                return True
            else:
                logger.error("❌ Excel file was not created")
                return False
        else:
            logger.error("❌ Excel export failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in Excel export test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run Excel export test."""
    logger.info("Starting Excel Export Test...")
    
    success = test_excel_export()
    
    if success:
        logger.info("🎉 Excel export test PASSED!")
        return True
    else:
        logger.warning("⚠️ Excel export test FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
