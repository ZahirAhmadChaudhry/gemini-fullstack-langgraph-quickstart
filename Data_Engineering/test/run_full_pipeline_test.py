#!/usr/bin/env python
"""
Full Pipeline Test for French Transcript Preprocessing

This script runs a comprehensive test of the entire preprocessing pipeline
on sample transcript files from multiple sources, including YouTube transcripts,
PDF documents, and DOCX files. It verifies:

1. Memory usage during processing
2. Correct handling of different file formats
3. Proper feature extraction
4. ML-ready output format compliance

Usage:
    python run_full_pipeline_test.py

Output:
    - Test results are saved to test/results/full_pipeline_test_results.json
    - Performance metrics are logged to test/logs/full_pipeline_test.log
"""

import os
import sys
import json
import time
import logging
import psutil
from pathlib import Path
from datetime import datetime

# Adjust Python path to allow importing from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import required modules
from utils.docx_processor import OptimizedDocxProcessor
from utils.pdf_processor import OptimizedPdfProcessor
from utils.encoding_detector import RobustEncodingDetector
from utils.sentence_tokenizer import ImprovedSentenceTokenizer
from utils.ml_formatter import MlReadyFormatter
import preprocess_transcripts as pipeline

# Configure logging
logging.basicConfig(
    filename=Path(__file__).parent / 'logs' / 'full_pipeline_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('full_pipeline_test')

class FullPipelineTest:
    """Test runner for the complete preprocessing pipeline."""
    
    def __init__(self):
        """Initialize test environment and resources."""
        self.test_dir = Path(__file__).parent
        self.sample_data_dir = self.test_dir / 'data'
        self.results_dir = self.test_dir / 'results'
        self.preprocessed_dir = self.test_dir / 'preprocessed_data'
        
        self.ensure_directories_exist()
        self.test_files = self.collect_test_files()
        
        # Initialize metrics dictionary
        self.metrics = {
            'overall': {
                'start_time': None,
                'end_time': None,
                'total_duration': None,
                'peak_memory_usage_mb': None
            },
            'file_metrics': {}
        }
    
    def ensure_directories_exist(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.results_dir, self.preprocessed_dir, self.test_dir / 'logs']:
            directory.mkdir(exist_ok=True, parents=True)
    
    def collect_test_files(self):
        """Collect test files for processing."""
        test_files = []
        
        # Check if we have sample files in the test directory
        if not list(self.sample_data_dir.glob('*')):
            # Copy some sample files from the main data directory for testing
            logger.info("No test files found. Copying sample files...")
            
            # If there are no files in test/data, copy some from the main data or files directory
            main_data_dir = Path(__file__).parent.parent / 'data'
            main_files_dir = Path(__file__).parent.parent / 'files'
            
            # Create a list of source directories to check
            source_dirs = [d for d in [main_data_dir, main_files_dir] if d.exists()]
            
            if not source_dirs:
                logger.error("No source directories found for test data")
                raise FileNotFoundError("No source directories found for test data")
            
            # Copy up to 5 files for testing
            file_count = 0
            for source_dir in source_dirs:
                for file_path in source_dir.glob('*'):
                    if file_path.is_file() and file_count < 5:
                        import shutil
                        shutil.copy(file_path, self.sample_data_dir / file_path.name)
                        test_files.append(self.sample_data_dir / file_path.name)
                        file_count += 1
                
                if file_count >= 5:
                    break
        else:
            # Use existing files in test/data
            test_files = list(self.sample_data_dir.glob('*'))
        
        logger.info(f"Collected {len(test_files)} files for testing")
        return test_files
    
    def track_memory(self):
        """Get current memory usage of the process in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    
    def run_test(self):
        """Run the full pipeline test on all collected files."""
        if not self.test_files:
            logger.error("No test files to process")
            return {"error": "No test files to process"}
        
        logger.info(f"Starting full pipeline test with {len(self.test_files)} files")
        
        # Track overall metrics
        self.metrics['overall']['start_time'] = datetime.now().isoformat()
        initial_memory = self.track_memory()
        peak_memory = initial_memory
        
        # Process each test file
        for file_path in self.test_files:
            file_name = file_path.name
            logger.info(f"Processing file: {file_name}")
            
            # Initialize file metrics
            self.metrics['file_metrics'][file_name] = {
                'start_time': datetime.now().isoformat(),
                'file_size_kb': file_path.stat().st_size / 1024,
                'file_type': file_path.suffix,
                'success': False,
                'error': None,
                'memory_before_mb': self.track_memory(),
                'memory_after_mb': None,
                'memory_diff_mb': None,
                'processing_time_sec': None
            }
            
            try:
                # Note the time before processing
                start_time = time.time()
                
                # Process the file using our pipeline
                output_file = self.preprocessed_dir / f"{file_path.stem}_ml_ready.json"
                  # Create preprocessor instance
                preprocessor = pipeline.TranscriptPreprocessor()
                
                # Create output directory if it doesn't exist
                output_dir = output_file.parent
                output_dir.mkdir(exist_ok=True, parents=True)
                
                # Process the individual file
                try:
                    # Process the file and get standard output
                    result = preprocessor.preprocess_transcript(file_path, file_path.name)
                    
                    # Save results manually to the specified location
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                        
                    logger.info(f"Saved test output to {output_file}")
                except Exception as e:
                    logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
                    raise
                
                # Record processing time
                processing_time = time.time() - start_time
                self.metrics['file_metrics'][file_name]['processing_time_sec'] = processing_time
                
                # Check memory usage after processing
                current_memory = self.track_memory()
                peak_memory = max(peak_memory, current_memory)
                self.metrics['file_metrics'][file_name]['memory_after_mb'] = current_memory
                self.metrics['file_metrics'][file_name]['memory_diff_mb'] = current_memory - self.metrics['file_metrics'][file_name]['memory_before_mb']
                
                # Verify the output file exists
                if output_file.exists():
                    # Validate the output format
                    with open(output_file, 'r', encoding='utf-8') as f:
                        output_data = json.load(f)
                    
                    # Check if the output has the expected structure
                    is_valid = self.validate_output(output_data)
                    
                    self.metrics['file_metrics'][file_name]['success'] = is_valid
                    self.metrics['file_metrics'][file_name]['output_size_kb'] = output_file.stat().st_size / 1024
                    
                    # Count segments and features
                    if 'segments' in output_data:
                        self.metrics['file_metrics'][file_name]['segment_count'] = len(output_data['segments'])
                    
                    logger.info(f"Processed {file_name} successfully in {processing_time:.2f} seconds")
                else:
                    self.metrics['file_metrics'][file_name]['error'] = "Output file not created"
                    logger.error(f"Failed to create output file for {file_name}")
            
            except Exception as e:
                self.metrics['file_metrics'][file_name]['error'] = str(e)
                logger.error(f"Error processing {file_name}: {e}", exc_info=True)
            
            # Force garbage collection to reduce memory fragmentation between files
            import gc
            gc.collect()
        
        # Complete overall metrics
        self.metrics['overall']['end_time'] = datetime.now().isoformat()
        self.metrics['overall']['peak_memory_usage_mb'] = peak_memory
        
        # Calculate duration
        start_dt = datetime.fromisoformat(self.metrics['overall']['start_time'])
        end_dt = datetime.fromisoformat(self.metrics['overall']['end_time'])
        self.metrics['overall']['total_duration'] = (end_dt - start_dt).total_seconds()
        
        # Save results
        self.save_results()
        
        logger.info(f"Full pipeline test completed. Peak memory usage: {peak_memory:.2f} MB")
        
        return self.metrics
    
    def validate_output(self, output_data):
        """Validate the structure of the output data."""
        try:
            # Check for required top-level keys
            required_keys = ['source_file', 'processed_timestamp', 'segments']
            for key in required_keys:
                if key not in output_data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            # Validate segments
            if not isinstance(output_data['segments'], list):
                logger.error(f"'segments' is not a list")
                return False
            
            if not output_data['segments']:
                logger.warning(f"'segments' list is empty")
                return True  # Empty but valid
            
            # Check the first segment for required structure
            first_segment = output_data['segments'][0]
            segment_keys = ['id', 'text', 'features', 'metadata']
            for key in segment_keys:
                if key not in first_segment:
                    logger.error(f"Segment missing required key: {key}")
                    return False
            
            # Validate features
            feature_keys = ['temporal_context', 'discourse_markers', 'sentence_count', 'word_count']
            for key in feature_keys:
                if key not in first_segment['features']:
                    logger.error(f"Features missing required key: {key}")
                    return False
            
            # Validate metadata
            metadata_keys = ['source', 'segment_lines', 'position']
            for key in metadata_keys:
                if key not in first_segment['metadata']:
                    logger.error(f"Metadata missing required key: {key}")
                    return False
            
            # All checks passed
            return True
        
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return False
    
    def save_results(self):
        """Save the test results and metrics."""
        results_path = self.results_dir / 'full_pipeline_test_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Saved test results to {results_path}")
        
        # Create a summary log
        success_count = sum(1 for file_metrics in self.metrics['file_metrics'].values() if file_metrics['success'])
        total_count = len(self.metrics['file_metrics'])
        
        summary = {
            'run_date': datetime.now().isoformat(),
            'success_rate': f"{success_count}/{total_count} ({success_count/total_count*100:.1f}%)",
            'total_duration': f"{self.metrics['overall']['total_duration']:.2f} seconds",
            'peak_memory': f"{self.metrics['overall']['peak_memory_usage_mb']:.2f} MB"
        }
        
        summary_path = self.results_dir / 'test_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

if __name__ == '__main__':
    try:
        test_runner = FullPipelineTest()
        results = test_runner.run_test()
        
        # Print summary to console
        print("\n=== FULL PIPELINE TEST RESULTS ===")
        success_count = sum(1 for file_metrics in results['file_metrics'].values() if file_metrics['success'])
        total_count = len(results['file_metrics'])
        print(f"Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print(f"Total duration: {results['overall']['total_duration']:.2f} seconds")
        print(f"Peak memory usage: {results['overall']['peak_memory_usage_mb']:.2f} MB")
        print(f"Detailed results saved to {test_runner.results_dir / 'full_pipeline_test_results.json'}")
        
        sys.exit(0 if success_count == total_count else 1)
    
    except Exception as e:
        logger.critical(f"Test runner failed: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)
