#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test case for ML integration components.

This test validates the integration of ML-ready formatting and YouTube transcript
handling in the transcript preprocessing pipeline.
"""

import os
import sys
import json
from pathlib import Path
import unittest
import logging
import importlib.util

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix numpy compatibility issues before importing spaCy
try:
    logger.info("Attempting to fix numpy compatibility...")
    import numpy
    logger.info(f"Current numpy version: {numpy.__version__}")
except ImportError:
    logger.warning("Numpy not found, installing compatible version...")
    import subprocess
    subprocess.call([sys.executable, "-m", "pip", "install", "numpy==1.24.3"])
    import numpy
    logger.info(f"Installed numpy version: {numpy.__version__}")

# Function to import a module from file path
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not find module {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Find project paths
project_root = Path(__file__).parent.parent.parent
logger.info(f"Project root: {project_root}")

# Import the utility modules directly from their file paths
ml_formatter_path = project_root / "utils" / "ml_formatter.py"
sentence_tokenizer_path = project_root / "utils" / "sentence_tokenizer.py"

try:
    # Import modules from file paths directly
    logger.info(f"Importing ML formatter from {ml_formatter_path}")
    ml_formatter_module = import_module_from_file("ml_formatter", ml_formatter_path)
    MlReadyFormatter = ml_formatter_module.MlReadyFormatter
    
    logger.info(f"Importing sentence tokenizer from {sentence_tokenizer_path}")
    sentence_tokenizer_module = import_module_from_file("sentence_tokenizer", sentence_tokenizer_path)
    ImprovedSentenceTokenizer = sentence_tokenizer_module.ImprovedSentenceTokenizer
    
    logger.info("Successfully imported utility classes")
except Exception as e:
    logger.error(f"Failed to import modules: {e}")
    raise ImportError(f"Could not import utility modules: {e}")

# Import spaCy now that numpy is compatible
try:
    import spacy
    logger.info("Successfully imported spaCy")
except ImportError:
    logger.error("Failed to import spaCy, attempting to install it")
    import subprocess
    subprocess.call([sys.executable, "-m", "pip", "install", "spacy==3.7.2"])
    import spacy


class TestMlIntegration(unittest.TestCase):
    """Tests for ML integration components."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Load a small spaCy model for testing
        try:
            cls.nlp = spacy.load("fr_core_news_sm")
        except:
            spacy.cli.download("fr_core_news_sm")
            cls.nlp = spacy.load("fr_core_news_sm")
            
        cls.test_dir = Path(__file__).parent
        cls.output_dir = cls.test_dir / "test_output"
        cls.output_dir.mkdir(exist_ok=True)
        
        # Sample data for testing
        cls.sample_segments = [
            {
                "segment_text": "Aujourd'hui, nous devons agir pour le climat.",
                "start_sentence_index": 0,
                "end_sentence_index": 1,
                "present_context": True,
                "future_context": False,
                "has_discourse_marker": True,
                "discourse_marker_type": "temporal"
            },
            {
                "segment_text": "En 2050, les énergies renouvelables seront dominantes.",
                "start_sentence_index": 2,
                "end_sentence_index": 3,
                "present_context": False,
                "future_context": True,
                "has_discourse_marker": False,
                "discourse_marker_type": ""
            }
        ]
    
    def test_ml_formatter(self):
        """Test ML formatter creates correct structure."""
        ml_formatter = MlReadyFormatter(str(self.output_dir))
        
        # Test formatting with sample data
        formatted_data = ml_formatter.format_segments(
            self.sample_segments,
            "test_file.txt"
        )
        
        # Validate basic structure
        self.assertIn("source_file", formatted_data)
        self.assertIn("processed_timestamp", formatted_data)
        self.assertIn("segments", formatted_data)
        self.assertEqual(len(formatted_data["segments"]), 2)
        
        # Validate segment structure
        segment = formatted_data["segments"][0]
        self.assertIn("id", segment)
        self.assertIn("text", segment)
        self.assertIn("features", segment)
        self.assertIn("metadata", segment)
        
        # Validate features
        self.assertIn("temporal_context", segment["features"])
        self.assertEqual(segment["features"]["temporal_context"], "2023")
        
        # Save and reload to test file saving
        output_path = ml_formatter.save_to_file(formatted_data, "test_ml_format.json")
        self.assertTrue(Path(output_path).exists())
        
        # Load and validate
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            self.assertEqual(len(loaded_data["segments"]), 2)
            
    def test_sentence_tokenizer_with_youtube(self):
        """Test improved sentence tokenizer handles YouTube transcripts."""
        tokenizer = ImprovedSentenceTokenizer(self.nlp)
        
        # Sample YouTube transcript text without proper punctuation
        youtube_text = """
        donc aujourd'hui je vais vous parler de l'avenir des énergies 
        nous allons examiner quelques projets innovants
        les voitures électriques représentent une solution intéressante 
        mais il y a encore des défis à surmonter
        regardons maintenant les statistiques récentes
        """
        
        # Test tokenization with aggressive mode (for YouTube)
        sentences = tokenizer.tokenize(youtube_text, aggressive=True)
        
        # Verify we get reasonable sentence boundaries
        self.assertTrue(len(sentences) >= 3)
        
        # Test with well-punctuated text
        normal_text = "Voici une première phrase. Et voici une deuxième. Enfin, une troisième."
        normal_sentences = tokenizer.tokenize(normal_text)
        self.assertEqual(len(normal_sentences), 3)


if __name__ == '__main__':
    unittest.main()
