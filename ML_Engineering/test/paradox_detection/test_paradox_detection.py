"""
Test script for the Paradox Detection module.

This script tests the functionality of the ParadoxDetector class, which implements
rule-based methods for detecting paradoxes in French sustainability discourse.

It includes tests for each detection method (antonym pairs, negated repetition,
contrastive structures, and sustainability tensions) as well as the overall paradox
detection functionality.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to sys.path to import the baseline_nlp package
current_dir = Path(__file__).parent
ml_engineering_dir = current_dir.parent.parent
sys.path.append(str(ml_engineering_dir))

from baseline_nlp.paradox_detection import ParadoxDetector
from baseline_nlp import config

class TestParadoxDetection(unittest.TestCase):
    """Test cases for paradox detection."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once before all test methods."""
        # Define paths to resource files
        data_dir = os.path.join(ml_engineering_dir, "baseline_nlp", "data")
        cls.sustainability_terms_path = os.path.join(data_dir, "sustainability_terms_fr.txt")
        cls.antonyms_path = os.path.join(data_dir, "french_antonyms.csv")
        cls.tension_keywords_path = os.path.join(data_dir, "tension_keywords_fr.csv")
        
        # Check if files exist
        for file_path in [cls.sustainability_terms_path, cls.antonyms_path, cls.tension_keywords_path]:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
        
        # Initialize detector with resources - use a lower confidence threshold for testing
        cls.detector = ParadoxDetector(
            sustainability_paradox_terms_path=cls.sustainability_terms_path,
            antonyms_path=cls.antonyms_path,
            tension_keywords_path=cls.tension_keywords_path,
            confidence_threshold=0.3  # Lower threshold for test purposes
        )
    
    def test_antonym_pair_detection(self):
        """Test detection of antonym pairs."""
        print("\nTesting antonym pair detection...")
        
        # Test with text containing antonym pairs
        text = "Nous devons augmenter la production tout en diminuant l'impact environnemental."
        doc = self.detector.nlp(text)
        
        # Debug: Print tokens and their lemmas
        print("Tokens and their lemmas:")
        for token in doc:
            print(f"{token.text}: {token.lemma_}")
        
        # Debug: Check if lemmas are in antonyms dictionary
        print("\nChecking if lemmas are in antonyms dictionary:")
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in self.detector.antonyms:
                print(f"Found {lemma} in antonyms dictionary")
                print(f"Antonyms for {lemma}: {self.detector.antonyms[lemma]}")
            
        # Debug: Use a simpler test case to verify antonym detection works at all
        simple_text = "augmenter diminuer"
        simple_doc = self.detector.nlp(simple_text)
        simple_detections = self.detector.detect_antonym_pairs(simple_doc)
        print("\nSimple test with 'augmenter diminuer':")
        print(f"Tokens: {[token.lemma_ for token in simple_doc]}")
        print(f"Detections: {simple_detections}")
        
        # Original test code
        detections = self.detector.detect_antonym_pairs(doc)
        
        # Debug: Print out the actual result before the assertion
        print(f"\nDetections result: {detections}")
        
        # Check if antonyms are detected
        self.assertTrue(detections, "Failed to detect antonyms in text")
        if detections:
            print(f"✓ Detected antonym pair: {detections[0]['word1']} and {detections[0]['word2']}")
        
        # Test with text without antonym pairs
        text_no_antonyms = "La durabilité est un concept important dans la société moderne."
        doc_no_antonyms = self.detector.nlp(text_no_antonyms)
        detections_no_antonyms = self.detector.detect_antonym_pairs(doc_no_antonyms)
        
        # Check that no antonyms are detected
        self.assertFalse(detections_no_antonyms, "False positive in antonym detection")
        if not detections_no_antonyms:
            print("✓ Correctly found no antonyms in text without antonym pairs")
    
    def test_negated_repetition_detection(self):
        """Test detection of negated repetition."""
        print("\nTesting negated repetition detection...")
        
        # Test with text containing negated repetition
        text = "Cette solution est durable et n'est pas durable en même temps."
        doc = self.detector.nlp(text)
        detections = self.detector.detect_negated_repetition(doc)
        
        # Check if negated repetition is detected
        self.assertTrue(detections, "Failed to detect negated repetition in text")
        if detections:
            print(f"✓ Detected negated repetition of word: {detections[0]['word']}")
        
        # Test with text without negated repetition
        text_no_negation = "Le développement durable est un concept important."
        doc_no_negation = self.detector.nlp(text_no_negation)
        detections_no_negation = self.detector.detect_negated_repetition(doc_no_negation)
        
        # Check that no negated repetition is detected
        self.assertFalse(detections_no_negation, "False positive in negated repetition detection")
        if not detections_no_negation:
            print("✓ Correctly found no negated repetition in text without negation")
    
    def test_contrastive_structures_detection(self):
        """Test detection of contrastive structures."""
        print("\nTesting contrastive structures detection...")
        
        # Test with text containing contrastive structures
        text = "D'une part, nous voulons augmenter la production, d'autre part, nous devons réduire les émissions de CO2."
        doc = self.detector.nlp(text)
        detections = self.detector.detect_contrastive_structures(doc)
        
        # Check if contrastive structures are detected
        self.assertTrue(detections, "Failed to detect contrastive structures in text")
        if detections:
            print(f"✓ Detected contrastive structure: {detections[0]['pattern']}")
        
        # Test with text without contrastive structures
        text_no_contrast = "Le développement durable est un concept important pour l'avenir."
        doc_no_contrast = self.detector.nlp(text_no_contrast)
        detections_no_contrast = self.detector.detect_contrastive_structures(doc_no_contrast)
        
        # Check that no contrastive structures are detected
        self.assertFalse(detections_no_contrast, "False positive in contrastive structures detection")
        if not detections_no_contrast:
            print("✓ Correctly found no contrastive structures in text without contrast")
    
    def test_sustainability_tensions_detection(self):
        """Test detection of sustainability tensions."""
        print("\nTesting sustainability tensions detection...")
        
        # Test with text containing sustainability tensions
        text = "Le développement durable crée un paradoxe entre la croissance économique et la protection de l'environnement."
        doc = self.detector.nlp(text)
        detections = self.detector.detect_sustainability_tensions(doc)
        
        # Check if sustainability tensions are detected
        self.assertTrue(detections, "Failed to detect sustainability tensions in text")
        if detections:
            print(f"✓ Detected sustainability tension with terms: {detections[0]['sustainability_terms']} and keywords: {detections[0]['tension_keywords']}")
        
        # Test with text without sustainability tensions
        text_no_tension = "L'économie mondiale continue de croître rapidement."
        doc_no_tension = self.detector.nlp(text_no_tension)
        detections_no_tension = self.detector.detect_sustainability_tensions(doc_no_tension)
        
        # Check that no sustainability tensions are detected
        self.assertFalse(detections_no_tension, "False positive in sustainability tensions detection")
        if not detections_no_tension:
            print("✓ Correctly found no sustainability tensions in text without sustainability terms and tension keywords")
    
    def test_overall_paradox_detection(self):
        """Test overall paradox detection."""
        print("\nTesting overall paradox detection...")
        
        # Test with text containing a clear paradox
        paradox_text = "Le développement durable exige que nous augmentions la production tout en réduisant l'impact environnemental. C'est un dilemme constant entre croissance économique et protection de l'environnement."
        result = self.detector.detect_paradoxes(paradox_text)
        
        # Check if paradox is detected
        self.assertTrue(result["is_paradox"], "Failed to detect paradox in text")
        if result["is_paradox"]:
            print(f"✓ Correctly detected paradox with confidence: {result['confidence']}")
            print(f"✓ Found {len(result['detections'])} supporting detections")
        
        # Test with text without paradox
        non_paradox_text = "Le développement durable est un concept important pour l'avenir de notre planète. Nous devons tous contribuer à un monde plus vert."
        result_no_paradox = self.detector.detect_paradoxes(non_paradox_text)
        
        # Check that no paradox is detected
        self.assertFalse(result_no_paradox["is_paradox"], "False positive in paradox detection")
        if not result_no_paradox["is_paradox"]:
            print(f"✓ Correctly identified non-paradoxical text with confidence: {result_no_paradox['confidence']}")
    
    def test_process_segment(self):
        """Test processing of a single segment."""
        print("\nTesting segment processing...")
        
        # Create a sample segment
        segment = {
            "text": ["Le développement durable pose un dilemme fondamental entre la croissance économique et la protection environnementale."],
            "segment_id": "test_segment_001"
        }
        
        # Process the segment
        processed_segment = self.detector.process_segment(segment)
        
        # Check that paradox field was added
        self.assertIn("paradox", processed_segment, "Failed to add paradox field to segment")
        if "paradox" in processed_segment:
            print(f"✓ Successfully added paradox detection to segment: {processed_segment['paradox']['is_paradox']}")
            print(f"✓ Confidence: {processed_segment['paradox']['confidence']}")
    
    def test_process_segments(self):
        """Test processing of multiple segments."""
        print("\nTesting multiple segment processing...")
        
        # Create sample segments
        segments = [
            {
                "text": ["Le développement durable pose un dilemme fondamental entre la croissance économique et la protection environnementale."],
                "segment_id": "test_segment_001"
            },
            {
                "text": ["La transition écologique est nécessaire pour l'avenir de notre planète."],
                "segment_id": "test_segment_002"
            }
        ]
        
        # Process the segments
        processed_segments = self.detector.process_segments(segments)
        
        # Check results
        self.assertEqual(len(processed_segments), 2, "Wrong number of processed segments")
        for segment in processed_segments:
            self.assertIn("paradox", segment, "Failed to add paradox field to segment")
        
        print(f"✓ Successfully processed {len(processed_segments)} segments")
        print(f"✓ First segment paradox: {processed_segments[0]['paradox']['is_paradox']}")
        print(f"✓ Second segment paradox: {processed_segments[1]['paradox']['is_paradox']}")

if __name__ == "__main__":
    unittest.main()