"""Test cases for encoding detection and handling."""

import unittest
from pathlib import Path
from typing import Dict, Any
import json

from ..config import TEST_DATA_DIR, ENCODINGS, logger
from ..utils import generate_test_file_with_encoding, create_french_text_with_markers
from preprocess_transcripts import TranscriptPreprocessor

class TestEncoding(unittest.TestCase):
    """Test cases for encoding detection and handling."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test files with different encodings."""
        cls.processor = TranscriptPreprocessor()
        cls.test_content = create_french_text_with_markers(
            num_segments=2,
            include_diacritics=True,
            include_discourse_markers=True,
            include_temporal_markers=True
        )
        cls.test_files = {}
        
        # Create test files with different encodings
        for encoding in ENCODINGS:
            filename = f"test_encoding_{encoding}.txt"
            filepath = generate_test_file_with_encoding(
                cls.test_content,
                filename,
                encoding
            )
            cls.test_files[encoding] = filepath
    
    def test_encoding_detection(self):
        """Test if encoding is correctly detected for each test file."""
        for encoding, filepath in self.test_files.items():
            detected_encoding = self.processor._detect_encoding(filepath)
            self.assertIsNotNone(detected_encoding,
                               f"Failed to detect encoding for {filepath}")
            logger.info(f"Detected encoding {detected_encoding} for {filepath}")
    
    def test_utf8_conversion(self):
        """Test if non-UTF-8 files are correctly converted to UTF-8."""
        for encoding, filepath in self.test_files.items():
            # Process the file
            result = self.processor.preprocess_transcript(filepath)
            
            # Verify no encoding errors in output
            self.assertIn("segments", result)
            
            # Check if diacritics are preserved
            text = " ".join([" ".join(seg["text"]) for seg in result["segments"]])
            self.assertIn("é", text, f"Diacritics not preserved in {filepath}")
            self.assertIn("è", text, f"Diacritics not preserved in {filepath}")
            
            # Save result for inspection
            output_path = TEST_DATA_DIR / f"{filepath.stem}_processed.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    
    def test_mojibake_correction(self):
        """Test correction of mojibake (garbled text)."""
        # Create file with simulated mojibake
        mojibake_text = self.test_content.encode('utf-8').decode('iso-8859-1').encode('utf-8')
        mojibake_file = generate_test_file_with_encoding(
            mojibake_text.decode('utf-8'),
            "test_mojibake.txt",
            'utf-8'
        )
        
        # Process and check if text is corrected
        result = self.processor.preprocess_transcript(mojibake_file)
        text = " ".join([" ".join(seg["text"]) for seg in result["segments"]])
        
        self.assertIn("é", text, "Mojibake not corrected - missing é")
        self.assertIn("è", text, "Mojibake not corrected - missing è")
        self.assertNotIn("Ã©", text, "Mojibake not corrected - found Ã©")
        self.assertNotIn("Ã¨", text, "Mojibake not corrected - found Ã¨")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        for filepath in cls.test_files.values():
            try:
                filepath.unlink()
                logger.info(f"Deleted test file {filepath}")
            except Exception as e:
                logger.error(f"Error deleting {filepath}: {e}")

if __name__ == '__main__':
    unittest.main()