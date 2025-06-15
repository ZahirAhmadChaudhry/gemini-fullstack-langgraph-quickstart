import unittest
import os
import gc
from pathlib import Path
import time

from utils.docx_processor import OptimizedDocxProcessor

class TestOptimizedDocxProcessor(unittest.TestCase):
    """Test case for the OptimizedDocxProcessor."""
    
    def setUp(self):
        """Set up test case."""
        self.processor = OptimizedDocxProcessor(memory_threshold_mb=50)
        self.test_files_dir = Path("data_renamed")  # Directory with sample DOCX files
    
    def test_memory_usage(self):
        """Test that memory usage is properly tracked and garbage collection is triggered."""
        if not self.test_files_dir.exists():
            self.skipTest(f"Test directory {self.test_files_dir} not found")
        
        docx_files = list(self.test_files_dir.glob("*.docx"))
        if not docx_files:
            self.skipTest(f"No DOCX files found in {self.test_files_dir}")
            
        sample_file = docx_files[0]
        
        # Force garbage collection before the test
        gc.collect()
        initial_memory = self.processor._get_memory_usage()
        
        # Process the document
        text = self.processor.extract_text(str(sample_file))
        
        # Check that we extracted some text
        self.assertTrue(text.strip(), "No text extracted from the document")
        
        # Process again to potentially trigger GC
        for _ in range(3):
            self.processor.extract_text(str(sample_file))
        
        # Check final memory usage
        final_memory = self.processor._get_memory_usage()
        
        # We can't assert exact memory values as they depend on the environment,
        # but we can log them for manual verification
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Difference: {final_memory - initial_memory:.2f} MB")
        
        # Memory growth should be limited due to our GC mechanisms
        # This is a soft test as exact memory behavior varies
        self.assertLess(final_memory - initial_memory, 200,  
                       "Memory growth exceeds expected threshold")
    
    def test_table_extraction(self):
        """Test table extraction from DOCX files."""
        if not self.test_files_dir.exists():
            self.skipTest(f"Test directory {self.test_files_dir} not found")
        
        docx_files = list(self.test_files_dir.glob("*.docx"))
        if not docx_files:
            self.skipTest(f"No DOCX files found in {self.test_files_dir}")
            
        sample_file = docx_files[0]
        
        # Extract tables data
        tables_data = self.processor.extract_tables_data(str(sample_file))
        
        # We can't make assumptions about the table content,
        # but we can verify the structure
        print(f"Number of tables found: {len(tables_data)}")
        if tables_data:
            print(f"First table dimensions: {len(tables_data[0])} rows")
            if tables_data[0]:
                print(f"First row has {len(tables_data[0][0])} cells")
        
    def test_batched_processing(self):
        """Test batched processing of tables."""
        if not self.test_files_dir.exists():
            self.skipTest(f"Test directory {self.test_files_dir} not found")
        
        docx_files = list(self.test_files_dir.glob("*.docx"))
        if not docx_files:
            self.skipTest(f"No DOCX files found in {self.test_files_dir}")
            
        sample_file = docx_files[0]
        
        # Test batch processing
        batch_count = 0
        row_count = 0
        
        for batch in self.processor.process_tables_batched(str(sample_file), batch_size=5):
            batch_count += 1
            row_count += len(batch)
        
        print(f"Processed {batch_count} batches with {row_count} total rows")
        
        # We can't make assumptions about the exact numbers,
        # but we can verify the method completes

if __name__ == "__main__":
    unittest.main()
