"""
Main module for the Baseline NLP System for French Sustainability Opinion Analysis.

This module integrates all components of the NLP pipeline and provides
functionality to process data and generate labeled datasets.
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import configuration
from . import config

# Import components
from .topic_identification import KeywordExtractor
from .opinion_detection import SentimentAnalyzer
from .paradox_detection import ParadoxDetector
from .temporal_context import TemporalContextAnalyzer
from .utils import DataLoader

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING["level"]),
    format=config.LOGGING["format"]
)
logger = logging.getLogger(__name__)

class NLPPipeline:
    """
    Main NLP pipeline integrating all analysis components.
    """
    
    def __init__(self):
        """
        Initialize the NLP pipeline with all components.
        """
        logger.info("Initializing NLP Pipeline...")
        
        # Initialize data loader
        self.data_loader = DataLoader(config.DATA_DIR)
        
        # Initialize topic identification component
        logger.info("Initializing Topic Identification component...")
        self.topic_identifier = KeywordExtractor(
            method=config.TOPIC_IDENTIFICATION["method"],
            num_keywords=config.TOPIC_IDENTIFICATION["num_keywords"],
            sustainability_terms_path=config.TOPIC_IDENTIFICATION["sustainability_terms_path"],
            spacy_model=config.NLP_PIPELINE["spacy_model"]
        )
        
        # Initialize opinion detection component
        logger.info("Initializing Opinion Detection component...")
        self.sentiment_analyzer = SentimentAnalyzer(
            method=config.OPINION_DETECTION["method"],
            lexicon_name=config.OPINION_DETECTION["lexicon"],
            lexicon_path=config.OPINION_DETECTION["french_sentiment_lexicon_path"],
            negation_handling=config.OPINION_DETECTION["negation_handling"],
            spacy_model=config.NLP_PIPELINE["spacy_model"]
        )
        
        # Initialize paradox detection component
        logger.info("Initializing Paradox Detection component...")
        self.paradox_detector = ParadoxDetector(
            sustainability_paradox_terms_path=config.PARADOX_DETECTION["sustainability_paradox_terms_path"],
            antonyms_path=config.PARADOX_DETECTION["antonyms_path"],
            tension_keywords_path=config.PARADOX_DETECTION["tension_keywords_path"],
            confidence_threshold=config.PARADOX_DETECTION["confidence_threshold"],
            spacy_model=config.NLP_PIPELINE["spacy_model"]
        )
        
        # Initialize temporal context distinction component
        logger.info("Initializing Temporal Context Distinction component...")
        self.temporal_analyzer = TemporalContextAnalyzer(
            present_markers_path=config.TEMPORAL_CONTEXT["present_markers_path"],
            future_markers_path=config.TEMPORAL_CONTEXT["future_markers_path"],
            verb_tense_patterns_path=config.TEMPORAL_CONTEXT["verb_tense_patterns_path"],
            spacy_model=config.NLP_PIPELINE["spacy_model"]
        )
        
        logger.info("NLP Pipeline initialized successfully")
    
    def process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process text segments through the entire NLP pipeline.
        
        Args:
            segments: List of text segments to analyze
            
        Returns:
            List of segments with added analysis metadata
        """
        if not segments:
            return []
            
        logger.info(f"Processing {len(segments)} segments through NLP pipeline...")
        
        # Apply each component sequentially
        # 1. Topic Identification
        segments = self.topic_identifier.process_segments(segments)
        
        # 2. Opinion Detection
        segments = self.sentiment_analyzer.process_segments(segments)
        
        # 3. Paradox Detection
        segments = self.paradox_detector.process_segments(segments)
        
        # 4. Temporal Context Distinction
        segments = self.temporal_analyzer.process_segments(segments)
        
        logger.info("Completed processing all segments")
        return segments
    
    def process_file(self, filename: str) -> Dict[str, Any]:
        """
        Process a single preprocessed file through the NLP pipeline.
        
        Args:
            filename: Name of the JSON file to process
            
        Returns:
            Dictionary with processed file data
        """
        logger.info(f"Processing file: {filename}")
        
        # Load data
        data = self.data_loader.load_file(filename)
        
        # Extract segments
        segments = self.data_loader.extract_segments(data)
        
        # Process segments
        processed_segments = self.process_segments(segments)
        
        # Update data with processed segments
        result = data.copy()
        result["segments"] = processed_segments
        result["processing_timestamp"] = datetime.datetime.now().isoformat()
        
        return result
    
    def process_all_files(self) -> List[Dict[str, Any]]:
        """
        Process all files in the data directory through the NLP pipeline.
        
        Returns:
            List of dictionaries with processed file data
        """
        logger.info("Processing all files...")
        
        # Load all data files
        data_files = self.data_loader.load_all_files()
        
        # Process each file
        processed_files = []
        for data in data_files:
            filename = data.get("filename", "unknown")
            logger.info(f"Processing file: {filename}")
            
            # Extract segments
            segments = self.data_loader.extract_segments(data)
            
            # Process segments
            processed_segments = self.process_segments(segments)
            
            # Update data with processed segments
            result = data.copy()
            result["segments"] = processed_segments
            result["processing_timestamp"] = datetime.datetime.now().isoformat()
            
            processed_files.append(result)
        
        logger.info(f"Processed {len(processed_files)} files")
        return processed_files
    
    def save_processed_data(self, data: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """
        Save processed data to JSON file.
        
        Args:
            data: Processed data to save
            output_dir: Directory to save output file (defaults to config.OUTPUT_DIR)
            
        Returns:
            Path to saved file
        """
        output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        filename = data.get("filename", "unknown")
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(output_dir, f"{base_name}_labeled.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed data to {output_path}")
        return output_path
    
    def save_all_processed_data(self, data_list: List[Dict[str, Any]], output_dir: Optional[str] = None) -> List[str]:
        """
        Save all processed data to JSON files.
        
        Args:
            data_list: List of processed data to save
            output_dir: Directory to save output files (defaults to config.OUTPUT_DIR)
            
        Returns:
            List of paths to saved files
        """
        output_paths = []
        for data in data_list:
            path = self.save_processed_data(data, output_dir)
            output_paths.append(path)
        
        return output_paths
    
    def generate_labeled_dataset(self) -> List[str]:
        """
        Generate labeled dataset by processing all files and saving results.
        
        Returns:
            List of paths to saved files
        """
        # Process all files
        processed_data = self.process_all_files()
        
        # Save processed data
        output_paths = self.save_all_processed_data(processed_data)
        
        return output_paths


def main():
    """
    Main entry point for running the NLP pipeline.
    """
    logger.info("Starting Baseline NLP System for French Sustainability Opinion Analysis")
    
    # Initialize pipeline
    pipeline = NLPPipeline()
    
    # Generate labeled dataset
    output_paths = pipeline.generate_labeled_dataset()
    
    logger.info(f"Saved labeled dataset to {len(output_paths)} files")
    logger.info("Completed successfully")


if __name__ == "__main__":
    main()

