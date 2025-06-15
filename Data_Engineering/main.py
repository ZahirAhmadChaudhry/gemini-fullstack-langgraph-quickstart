#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Entry Point for French Transcript Preprocessing Pipeline

This is the primary entry point for the enhanced French transcript preprocessing pipeline.
Provides command-line interface and configuration management for all pipeline operations.

Usage:
    python main.py [options]
    python main.py --config config_production.json
    python main.py --mode development --input data --output results
    python main.py --help

Author: Enhanced French Transcript Preprocessing Pipeline
Version: 2.0.0
Date: 2025-06-12
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, ProcessingMode, load_config
from preprocess_transcripts import TranscriptPreprocessor

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced French Transcript Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default production configuration
  python main.py
  
  # Run in development mode with custom input/output
  python main.py --mode development --input data --output results
  
  # Use custom configuration file
  python main.py --config config_custom.json
  
  # Process specific files only
  python main.py --files transcript1.txt transcript2.docx
  
  # Enable parallel processing
  python main.py --parallel --workers 8
  
  # Validate configuration without processing
  python main.py --validate-config
  
  # Generate sample configuration file
  python main.py --generate-config sample_config.json
        """
    )
    
    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to JSON configuration file"
    )
    config_group.add_argument(
        "--mode", "-m",
        type=str,
        choices=["development", "production", "testing", "benchmark"],
        default="production",
        help="Processing mode (default: production)"
    )
    config_group.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    config_group.add_argument(
        "--generate-config",
        type=str,
        metavar="FILE",
        help="Generate sample configuration file and exit"
    )
    
    # Input/Output options
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--input", "-i",
        type=str,
        default="data",
        help="Input directory or file path (default: data)"
    )
    io_group.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory (overrides config)"
    )
    io_group.add_argument(
        "--files", "-f",
        nargs="+",
        help="Specific files to process (relative to input directory)"
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing"
    )
    proc_group.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers"
    )
    proc_group.add_argument(
        "--segmentation",
        type=str,
        choices=["auto", "sentence_based", "word_based", "hybrid"],
        help="Segmentation strategy"
    )
    
    # Output format options
    format_group = parser.add_argument_group("Output Formats")
    format_group.add_argument(
        "--no-standard",
        action="store_true",
        help="Disable standard output format"
    )
    format_group.add_argument(
        "--no-ml-ready",
        action="store_true",
        help="Disable ML-ready output format"
    )
    format_group.add_argument(
        "--no-target-format",
        action="store_true",
        help="Disable target format output"
    )
    
    # Logging options
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    log_group.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    log_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output"
    )
    log_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Utility options
    util_group = parser.add_argument_group("Utilities")
    util_group.add_argument(
        "--version",
        action="version",
        version="Enhanced French Transcript Preprocessing Pipeline v2.0.0"
    )
    util_group.add_argument(
        "--list-files",
        action="store_true",
        help="List files that would be processed and exit"
    )
    util_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually processing"
    )
    
    return parser

def apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace):
    """Apply command-line argument overrides to configuration."""
    
    # Output directory override
    if args.output:
        config.output.base_output_dir = args.output
    
    # Parallel processing overrides
    if args.parallel:
        config.processing.enable_parallel_processing = True
    if args.workers:
        config.processing.max_workers = args.workers
    
    # Segmentation strategy override
    if args.segmentation:
        from config import SegmentationStrategy
        config.segmentation.strategy = SegmentationStrategy(args.segmentation)
    
    # Output format overrides
    if args.no_standard:
        config.output.create_standard_output = False
    if args.no_ml_ready:
        config.output.create_ml_ready_output = False
    if args.no_target_format:
        config.output.create_target_format_output = False
    
    # Logging overrides
    if args.log_level:
        config.logging.level = args.log_level
    if args.log_file:
        config.logging.file_path = args.log_file
    if args.quiet:
        config.logging.enable_console_output = False
    if args.verbose:
        config.logging.level = "DEBUG"

def validate_configuration(config: PipelineConfig) -> bool:
    """Validate configuration and print any issues."""
    issues = config.validate()
    
    if issues:
        print("Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Configuration validation passed.")
        return True

def list_input_files(input_path: str, specific_files: Optional[List[str]] = None) -> List[Path]:
    """List files that would be processed."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        return [input_path]
    
    if not input_path.is_dir():
        print(f"Error: Input path '{input_path}' does not exist.")
        return []
    
    files = []
    supported_extensions = [".txt", ".docx", ".pdf", ".json"]
    
    if specific_files:
        for file_name in specific_files:
            file_path = input_path / file_name
            if file_path.exists():
                files.append(file_path)
            else:
                print(f"Warning: File '{file_path}' not found.")
    else:
        for ext in supported_extensions:
            files.extend(input_path.glob(f"*{ext}"))
    
    return sorted(files)

def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Handle utility commands first
        if args.generate_config:
            config = PipelineConfig()
            config.save_to_file(args.generate_config)
            print(f"Sample configuration saved to: {args.generate_config}")
            return 0
        
        # Load configuration
        if args.config:
            if not os.path.exists(args.config):
                print(f"Error: Configuration file '{args.config}' not found.")
                return 1
            config = load_config(args.config, ProcessingMode(args.mode))
        else:
            config = PipelineConfig(mode=ProcessingMode(args.mode))
        
        # Apply CLI overrides
        apply_cli_overrides(config, args)
        
        # Validate configuration
        if args.validate_config:
            return 0 if validate_configuration(config) else 1
        
        if not validate_configuration(config):
            return 1
        
        # Setup logging
        logger = config.setup_logging()
        
        # List files if requested
        if args.list_files:
            files = list_input_files(args.input, args.files)
            print(f"Files to be processed ({len(files)}):")
            for file_path in files:
                print(f"  - {file_path}")
            return 0
        
        # Dry run
        if args.dry_run:
            files = list_input_files(args.input, args.files)
            print("Dry run - would process the following:")
            print(f"  Input: {args.input}")
            print(f"  Output: {config.output.base_output_dir}")
            print(f"  Files: {len(files)}")
            print(f"  Mode: {config.mode.value}")
            print(f"  Parallel: {config.processing.enable_parallel_processing}")
            return 0
        
        # Create output directories
        paths = config.get_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize and run preprocessor
        logger.info("Starting Enhanced French Transcript Preprocessing Pipeline v2.0.0")
        logger.info(f"Mode: {config.mode.value}")
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {config.output.base_output_dir}")

        # Initialize preprocessor (no parameters needed)
        preprocessor = TranscriptPreprocessor()

        # Process files using the existing interface
        if args.files:
            # Process specific files
            for file_name in args.files:
                file_path = Path(args.input) / file_name
                if file_path.exists():
                    logger.info(f"Processing specific file: {file_path}")
                    # Generate file ID from filename
                    file_id = file_path.stem
                    # Use the existing preprocess_transcript method
                    preprocessor.preprocess_transcript(file_path, file_id)
                    logger.info(f"Successfully processed: {file_path}")
                else:
                    logger.warning(f"File not found: {file_path}")
        else:
            # Process all files in directory using existing method
            logger.info(f"Processing all files in directory: {args.input}")
            input_path = Path(args.input)
            output_path = Path(config.output.base_output_dir)
            # Use the existing preprocess_all method
            preprocessor.preprocess_all(input_path, output_path)
        
        logger.info("Pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
