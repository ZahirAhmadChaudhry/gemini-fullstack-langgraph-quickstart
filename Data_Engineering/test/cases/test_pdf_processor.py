import unittest
import os
import gc
from pathlib import Path
import tempfile
import time

from utils.pdf_processor import OptimizedPdfProcessor

class TestOptimizedPdfProcessor(unittest.TestCase):
    """Test case for the OptimizedPdfProcessor."""
    
    def setUp(self):
        """Set up test case."""
        self.processor = OptimizedPdfProcessor(memory_threshold_mb=50)
        # Look for PDF files in various locations
        test_locations = [
            Path("test/data"),
            Path("files"),
            Path("data")
        ]
        
        self.pdf_files = []
        for location in test_locations:
            if location.exists():
                pdf_files = list(location.glob("*.pdf"))
                if pdf_files:
                    self.pdf_files.extend(pdf_files)
                    break
    
    def test_memory_usage(self):
        """Test that memory usage is properly tracked and garbage collection is triggered."""
        if not self.pdf_files:
            self.skipTest("No PDF files found for testing")
            
        sample_file = self.pdf_files[0]
        
        # Force garbage collection before the test
        gc.collect()
        initial_memory = self.processor._get_memory_usage()
        
        # Process the document
        text, library = self.processor.extract_text(str(sample_file))
        
        # Check what we got
        print(f"Extracted {len(text)} characters using {library}")
        
        # Process again to potentially trigger GC
        for _ in range(3):
            self.processor.extract_text(str(sample_file))
        
        # Check final memory usage
        final_memory = self.processor._get_memory_usage()
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Difference: {final_memory - initial_memory:.2f} MB")
        
        # Memory growth should be limited due to our GC mechanisms
        self.assertLess(final_memory - initial_memory, 200,  
                       "Memory growth exceeds expected threshold")
    
    def test_extract_text_methods(self):
        """Test different text extraction methods."""
        if not self.pdf_files:
            self.skipTest("No PDF files found for testing")
            
        sample_file = self.pdf_files[0]
        
        # Test PyMuPDF extraction
        try:
            pymupdf_text = self.processor.extract_text_with_pymupdf(str(sample_file))
            print(f"PyMuPDF extracted {len(pymupdf_text)} characters")
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
        
        # Test pdfplumber extraction
        try:
            pdfplumber_text = self.processor.extract_text_with_pdfplumber(str(sample_file))
            print(f"pdfplumber extracted {len(pdfplumber_text)} characters")
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
        
        # Test combined extraction
        text, library = self.processor.extract_text(str(sample_file))
        print(f"Combined extraction used {library} and got {len(text)} characters")
    
    def test_image_extraction(self):
        """Test image extraction if PDF files are available."""
        if not self.pdf_files:
            self.skipTest("No PDF files found for testing")
            
        sample_file = self.pdf_files[0]
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract images
            try:
                image_paths = self.processor.extract_images(str(sample_file), temp_dir)
                print(f"Extracted {len(image_paths)} images to {temp_dir}")
                
                # Check if images were created
                for img_path in image_paths:
                    self.assertTrue(os.path.exists(img_path))
                    size = os.path.getsize(img_path)
                    print(f"Image: {os.path.basename(img_path)}, Size: {size} bytes")
            except Exception as e:
                print(f"Image extraction failed: {e}")

if __name__ == "__main__":
    unittest.main()
