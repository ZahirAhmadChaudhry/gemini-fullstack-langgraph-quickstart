"""
Test script for the Temporal Context Analysis module.

This script tests the functionality of the TemporalContextAnalyzer class, which implements
rule-based methods for distinguishing between present (2023) and future (2050) contexts
in French sustainability discourse.

It includes tests for temporal marker detection, verb tense detection, and overall 
temporal context analysis.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to sys.path to import the baseline_nlp package
current_dir = Path(__file__).parent
ml_engineering_dir = current_dir.parent.parent
sys.path.append(str(ml_engineering_dir))

from baseline_nlp.temporal_context import TemporalContextAnalyzer
from baseline_nlp import config

class TestTemporalContextAnalysis(unittest.TestCase):
    """Test cases for temporal context analysis."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once before all test methods."""
        # Define paths to resource files
        data_dir = os.path.join(ml_engineering_dir, "baseline_nlp", "data")
        cls.present_markers_path = os.path.join(data_dir, "present_markers_fr.csv")
        cls.future_markers_path = os.path.join(data_dir, "future_markers_fr.csv")
        
        # The verb tense patterns path may not exist yet, so we'll use default patterns
        cls.verb_tense_patterns_path = None
        
        # Check if files exist
        for file_path in [cls.present_markers_path, cls.future_markers_path]:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
        
        # Initialize analyzer with resources
        cls.analyzer = TemporalContextAnalyzer(
            present_markers_path=cls.present_markers_path,
            future_markers_path=cls.future_markers_path,
            verb_tense_patterns_path=cls.verb_tense_patterns_path
        )
    
    def test_temporal_markers_detection(self):
        """Test detection of temporal markers."""
        print("\nTesting temporal markers detection...")
        
        # Test with text containing present markers
        present_text = "Aujourd'hui en 2023, le développement durable est un concept important dans la société moderne."
        present_markers = self.analyzer.detect_temporal_markers(present_text)
        
        # Check if present markers are detected
        self.assertTrue(present_markers["present"], "Failed to detect present markers in text")
        self.assertFalse(present_markers["future"], "False positive in future marker detection")
        if present_markers["present"]:
            print(f"✓ Detected present markers: {present_markers['present']}")
        
        # Test with text containing future markers
        future_text = "En 2050, le développement durable sera au cœur des politiques environnementales."
        future_markers = self.analyzer.detect_temporal_markers(future_text)
        
        # Check if future markers are detected
        self.assertTrue(future_markers["future"], "Failed to detect future markers in text")
        if future_markers["future"]:
            print(f"✓ Detected future markers: {future_markers['future']}")
        
        # Test with text containing both present and future markers
        mixed_text = "Aujourd'hui nous prenons des mesures pour que demain, en 2050, notre planète soit préservée."
        mixed_markers = self.analyzer.detect_temporal_markers(mixed_text)
        
        # Check if both types of markers are detected
        self.assertTrue(mixed_markers["present"], "Failed to detect present markers in mixed text")
        self.assertTrue(mixed_markers["future"], "Failed to detect future markers in mixed text")
        if mixed_markers["present"] and mixed_markers["future"]:
            print(f"✓ Detected present markers in mixed text: {mixed_markers['present']}")
            print(f"✓ Detected future markers in mixed text: {mixed_markers['future']}")
    
    def test_verb_tense_detection(self):
        """Test detection of verb tenses."""
        print("\nTesting verb tense detection...")
        
        # Test with text containing present tense verbs
        present_text = "Nous travaillons sur le développement durable. Les entreprises adoptent des pratiques écologiques."
        present_doc = self.analyzer.nlp(present_text)
        tense_counts = self.analyzer.detect_verb_tenses(present_doc)
        
        # Check if present tense is detected
        self.assertIn("present", tense_counts, "Failed to detect present tense in text")
        if "present" in tense_counts:
            print(f"✓ Detected present tense verbs: {tense_counts['present']} occurrences")
        
        # Test with text containing future tense verbs
        future_text = "Nous adopterons des politiques environnementales plus strictes. Les entreprises devront s'adapter."
        future_doc = self.analyzer.nlp(future_text)
        future_tense_counts = self.analyzer.detect_verb_tenses(future_doc)
        
        # Check if future tense is detected
        self.assertTrue(
            any(tense in future_tense_counts for tense in ["futur_simple", "futur_proche", "conditionnel"]),
            "Failed to detect future tense in text"
        )
        print(f"✓ Detected verb tenses in future text: {future_tense_counts}")
        
        # Test with text containing present tense used for future
        present_for_future_text = "Demain, nous partons en mission pour sauver la planète."
        pf_doc = self.analyzer.nlp(present_for_future_text)
        
        # First get raw tense counts
        pf_tense_counts = self.analyzer.detect_verb_tenses(pf_doc)
        
        # Then disambiguate present tense usage
        disambiguated_counts = self.analyzer.disambiguate_present_tense(pf_doc, pf_tense_counts)
        
        # Check if present-for-future is detected
        self.assertIn("present_for_future", disambiguated_counts, 
                    "Failed to disambiguate present-for-future usage")
        if "present_for_future" in disambiguated_counts:
            print(f"✓ Detected present-for-future usage: {disambiguated_counts['present_for_future']} occurrences")
    
    def test_temporal_context_analysis(self):
        """Test overall temporal context analysis."""
        print("\nTesting overall temporal context analysis...")
        
        # Test with clearly present-oriented text
        present_text = "Aujourd'hui en 2023, nous constatons une prise de conscience accrue sur les enjeux environnementaux. Les entreprises adoptent progressivement des pratiques durables, mais le chemin est encore long."
        present_result = self.analyzer.analyze_temporal_context(present_text)
        
        # Check if context is identified as present
        self.assertEqual(present_result["context"], "present", 
                        "Failed to identify present context")
        print(f"✓ Correctly identified present context with confidence: {present_result['confidence']}")
        
        # Test with clearly future-oriented text
        future_text = "En 2050, nous aurons transformé notre économie pour qu'elle soit entièrement circulaire. Les entreprises adopteront des pratiques zéro déchet, et les énergies renouvelables domineront le marché."
        future_result = self.analyzer.analyze_temporal_context(future_text)
        
        # Check if context is identified as future
        self.assertEqual(future_result["context"], "future", 
                        "Failed to identify future context")
        print(f"✓ Correctly identified future context with confidence: {future_result['confidence']}")
        
        # Test with ambiguous text (no clear temporal markers)
        ambiguous_text = "Le développement durable représente un défi majeur pour notre société. Il faut repenser nos modèles économiques."
        ambiguous_result = self.analyzer.analyze_temporal_context(ambiguous_text)
        
        # Print the result for ambiguous text
        print(f"✓ Analysis for ambiguous text: context={ambiguous_result['context']}, confidence={ambiguous_result['confidence']}")
    
    def test_process_segment(self):
        """Test processing of a single segment."""
        print("\nTesting segment processing...")
        
        # Create a sample segment with present context
        present_segment = {
            "text": ["Aujourd'hui, les entreprises françaises sont confrontées à de nombreux défis environnementaux."],
            "segment_id": "test_segment_001"
        }
        
        # Process the segment
        processed_present = self.analyzer.process_segment(present_segment)
        
        # Check that temporal_context field was added
        self.assertIn("temporal_context", processed_present, 
                    "Failed to add temporal_context field to segment")
        self.assertEqual(processed_present["temporal_context"]["context"], "present",
                        "Failed to identify present context in segment")
        print(f"✓ Successfully added temporal context to present segment: {processed_present['temporal_context']['context']}")
        print(f"✓ Confidence: {processed_present['temporal_context']['confidence']}")
        
        # Create a sample segment with future context
        future_segment = {
            "text": ["En 2050, les entreprises françaises auront adopté des modèles d'affaires entièrement durables."],
            "segment_id": "test_segment_002"
        }
        
        # Process the segment
        processed_future = self.analyzer.process_segment(future_segment)
        
        # Check that temporal_context field was added with future context
        self.assertIn("temporal_context", processed_future, 
                    "Failed to add temporal_context field to segment")
        self.assertEqual(processed_future["temporal_context"]["context"], "future",
                        "Failed to identify future context in segment")
        print(f"✓ Successfully added temporal context to future segment: {processed_future['temporal_context']['context']}")
        print(f"✓ Confidence: {processed_future['temporal_context']['confidence']}")
    
    def test_process_segments(self):
        """Test processing of multiple segments."""
        print("\nTesting multiple segment processing...")
        
        # Create sample segments with different temporal contexts
        segments = [
            {
                "text": ["Aujourd'hui, les entreprises françaises sont confrontées à de nombreux défis environnementaux."],
                "segment_id": "test_segment_001"
            },
            {
                "text": ["En 2050, les entreprises françaises auront adopté des modèles d'affaires entièrement durables."],
                "segment_id": "test_segment_002"
            },
            {
                "text": ["Le développement durable est un concept qui évolue constamment."],
                "segment_id": "test_segment_003"
            }
        ]
        
        # Process the segments
        processed_segments = self.analyzer.process_segments(segments)
        
        # Check results
        self.assertEqual(len(processed_segments), 3, "Wrong number of processed segments")
        for segment in processed_segments:
            self.assertIn("temporal_context", segment, 
                        "Failed to add temporal_context field to segment")
        
        print(f"✓ Successfully processed {len(processed_segments)} segments")
        print(f"✓ First segment (present) context: {processed_segments[0]['temporal_context']['context']}")
        print(f"✓ Second segment (future) context: {processed_segments[1]['temporal_context']['context']}")
        print(f"✓ Third segment (ambiguous) context: {processed_segments[2]['temporal_context']['context']}")

if __name__ == "__main__":
    unittest.main()