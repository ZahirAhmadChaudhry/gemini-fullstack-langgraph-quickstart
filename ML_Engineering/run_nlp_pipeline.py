#!/usr/bin/env python
"""
Entry point script for running the Baseline NLP System for French Sustainability Opinion Analysis.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to sys.path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from baseline_nlp.main import NLPPipeline
import baseline_nlp.config as config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Baseline NLP System for French Sustainability Opinion Analysis')
    
    parser.add_argument('--file', '-f', 
                        help='Process a specific file instead of all files')
    
    parser.add_argument('--output-dir', '-o',
                        help='Directory to save output files')
    
    parser.add_argument('--log-level', '-l',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO',
                        help='Set the logging level')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Update output directory if provided
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        logger.info(f"Output directory set to: {config.OUTPUT_DIR}")
    
    try:
        # Initialize pipeline
        pipeline = NLPPipeline()
        
        # Process files
        if args.file:
            # Process a single file
            result = pipeline.process_file(args.file)
            output_path = pipeline.save_processed_data(result)
            logger.info(f"Processed {args.file} and saved to {output_path}")
        else:
            # Process all files
            output_paths = pipeline.generate_labeled_dataset()
            logger.info(f"Processed all files and saved to {len(output_paths)} output files")
        
        logger.info("Processing completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error in NLP pipeline: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())