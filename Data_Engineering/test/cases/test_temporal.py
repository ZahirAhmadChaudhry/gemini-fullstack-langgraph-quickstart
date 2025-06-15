"""Test cases for temporal marker detection and classification."""

import unittest
from pathlib import Path
from typing import Dict, Any
import json

from ..config import TEST_DATA_DIR, logger
from ..utils import generate_test_file_with_encoding
from preprocess_transcripts import TranscriptPreprocessor

class TestTemporal(unittest.TestCase):
    """Test cases for temporal marker detection and classification."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.processor = TranscriptPreprocessor()
        
        # Create test cases for explicit temporal references
        cls.explicit_2023_cases = [
            "En 2023, nous observons plusieurs changements.",
            "La situation actuelle en 2023 montre...",
            "Aujourd'hui, nous constatons...",
            "Actuellement, les données indiquent...",
            "Dans le présent, nous voyons..."
        ]
        
        cls.explicit_2050_cases = [
            "D'ici 2050, nous prévoyons...",
            "En 2050, la situation sera différente.",
            "Dans le futur, notamment en 2050...",
            "À l'horizon 2050, nous anticipons...",
            "L'avenir, en 2050, nous réserve..."
        ]
        
        # Create test cases for implicit temporal references
        cls.implicit_2023_cases = [
            "Nous observons actuellement une tendance...",
            "Les données montrent maintenant que...",
            "À l'heure actuelle, nous constatons...",
            "La situation présente indique...",
            "Nous remarquons aujourd'hui..."
        ]
        
        cls.implicit_2050_cases = [
            "Les changements transformeront notre façon de vivre.",
            "Cette technologie évoluera considérablement.",
            "Nous adopterons de nouvelles solutions.",
            "La société s'adaptera à ces changements.",
            "Ces innovations auront un impact majeur."
        ]
        
        # Create test cases for mixed temporal references
        cls.mixed_cases = [
            "En 2023, nous planifions les changements qui surviendront en 2050.",
            "Actuellement, nous développons les technologies qui seront cruciales en 2050.",
            "La situation présente nous permet de préparer l'avenir.",
            "Les données de 2023 nous aident à projeter les tendances futures."
        ]
    
    def test_explicit_2023_detection(self):
        """Test detection of explicit 2023 references."""
        for case in self.explicit_2023_cases:
            markers = self.processor._identify_temporal_markers([case])
            self.assertTrue(
                markers["2023_reference"],
                f"Failed to detect 2023 reference in: {case}"
            )
            self.assertFalse(
                markers["2050_reference"],
                f"Incorrectly detected 2050 reference in: {case}"
            )
    
    def test_explicit_2050_detection(self):
        """Test detection of explicit 2050 references."""
        for case in self.explicit_2050_cases:
            markers = self.processor._identify_temporal_markers([case])
            self.assertTrue(
                markers["2050_reference"],
                f"Failed to detect 2050 reference in: {case}"
            )
            self.assertFalse(
                markers["2023_reference"],
                f"Incorrectly detected 2023 reference in: {case}"
            )
    
    def test_implicit_2023_detection(self):
        """Test detection of implicit present/2023 references."""
        for case in self.implicit_2023_cases:
            markers = self.processor._identify_temporal_markers([case])
            self.assertTrue(
                markers["2023_reference"],
                f"Failed to detect implicit 2023 reference in: {case}"
            )
    
    def test_implicit_2050_detection(self):
        """Test detection of implicit future/2050 references."""
        for case in self.implicit_2050_cases:
            markers = self.processor._identify_temporal_markers([case])
            self.assertTrue(
                markers["2050_reference"],
                f"Failed to detect implicit 2050 reference in: {case}"
            )
    
    def test_mixed_temporal_references(self):
        """Test handling of text with both 2023 and 2050 references."""
        for case in self.mixed_cases:
            markers = self.processor._identify_temporal_markers([case])
            self.assertTrue(
                markers["2023_reference"],
                f"Failed to detect 2023 reference in mixed case: {case}"
            )
            self.assertTrue(
                markers["2050_reference"],
                f"Failed to detect 2050 reference in mixed case: {case}"
            )
    
    def test_verb_tense_detection(self):
        """Test detection of temporal references through verb tenses."""
        future_tense_cases = [
            "Nous développerons de nouvelles solutions.",
            "Les technologies évolueront rapidement.",
            "La société s'adaptera aux changements."
        ]
        
        for case in future_tense_cases:
            markers = self.processor._identify_temporal_markers([case])
            self.assertTrue(
                markers["2050_reference"],
                f"Failed to detect future reference from verb tense in: {case}"
            )
    
    def test_temporal_marker_persistence(self):
        """Test if temporal markers are correctly preserved in preprocessed output."""
        # Create a test file with mixed temporal references
        test_content = "\n\n".join(self.mixed_cases)
        test_file = generate_test_file_with_encoding(
            test_content,
            "test_temporal_persistence.txt",
            'utf-8'
        )
        
        # Process the file
        result = self.processor.preprocess_transcript(test_file)
        
        # Save result for inspection
        output_path = TEST_DATA_DIR / "temporal_persistence_processed.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # Check if temporal markers are preserved in output
        for segment in result["segments"]:
            self.assertIn("temporal_markers", segment,
                         "Temporal markers not preserved in output")
            self.assertIn("2023_reference", segment["temporal_markers"],
                         "2023 reference marker missing")
            self.assertIn("2050_reference", segment["temporal_markers"],
                         "2050 reference marker missing")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        files_to_clean = [
            "test_temporal_persistence.txt",
            "temporal_persistence_processed.json"
        ]
        
        for filename in files_to_clean:
            try:
                filepath = TEST_DATA_DIR / filename
                if filepath.exists():
                    filepath.unlink()
                    logger.info(f"Deleted test file {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up {filename}: {e}")

if __name__ == '__main__':
    unittest.main()