#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized PDF Processor Module

This module provides memory-optimized handling for PDF files,
addressing memory leaks in PyMuPDF (fitz) and other PDF libraries
when processing large documents.
"""

import fitz  # PyMuPDF
import pdfplumber
import gc
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any, Generator, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class OptimizedPdfProcessor:
    """
    Memory-optimized processor for PDF files.
    
    Addresses the memory leaks in PyMuPDF by implementing proper resource
    management, batch processing, and explicit garbage collection.
    """
    
    def __init__(self, memory_threshold_mb: int = 100):
        """
        Initialize the processor.
        
        Args:
            memory_threshold_mb: Memory threshold in MB after which to trigger garbage collection
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.initial_memory_usage = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            # Convert bytes to MB
            return memory_info.rss / (1024 * 1024)
        except ImportError:
            logger.warning("psutil not installed. Memory monitoring disabled.")
            return 0
    
    def _check_memory_usage(self):
        """Check current memory usage and trigger GC if above threshold."""
        current_memory = self._get_memory_usage()
        initial_memory = self.initial_memory_usage
        
        # Log memory usage
        logger.debug(f"Memory usage: {current_memory:.2f} MB")
        
        # If memory usage has increased beyond threshold, force garbage collection
        if current_memory - initial_memory > self.memory_threshold_mb:
            logger.info(f"Memory threshold exceeded. Triggering garbage collection.")
            gc.collect()
            new_memory = self._get_memory_usage()
            logger.info(f"Memory after GC: {new_memory:.2f} MB (freed {current_memory - new_memory:.2f} MB)")
    
    @contextmanager
    def _open_pymupdf(self, pdf_path: str):
        """
        Context manager for safely opening a PDF with PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Yields:
            The PDF document object
        """
        document = None
        try:
            document = fitz.open(pdf_path)
            yield document
        finally:
            # Manually close the document to prevent memory leaks
            if document:
                document.close()
            document = None
            # Force garbage collection
            gc.collect()
            self._check_memory_usage()
    
    @contextmanager
    def _open_pdfplumber(self, pdf_path: str):
        """
        Context manager for safely opening a PDF with pdfplumber.
        
        Args:
            pdf_path: Path to the PDF file
            
        Yields:
            The PDF document object
        """
        document = None
        try:
            document = pdfplumber.open(pdf_path)
            yield document
        finally:
            # Manually close the document to prevent memory leaks
            if document:
                document.close()
            document = None
            # Force garbage collection
            gc.collect()
            self._check_memory_usage()
    
    def extract_text_with_pymupdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF with memory management.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Reading PDF with optimized PyMuPDF: {pdf_path}")
        text_parts = []
        
        with self._open_pymupdf(pdf_path) as pdf:
            # Get number of pages
            num_pages = len(pdf)
            logger.info(f"PDF has {num_pages} pages")
            
            # Process pages in batches to manage memory
            batch_size = 5  # Process 5 pages at a time
            for i in range(0, num_pages, batch_size):
                end_idx = min(i + batch_size, num_pages)
                logger.debug(f"Processing pages {i+1} to {end_idx} of {num_pages}")
                
                # Extract text from each page in batch
                for page_num in range(i, end_idx):
                    try:
                        page = pdf[page_num]
                        page_text = page.get_text()
                        text_parts.append(page_text)
                    except Exception as e:
                        logger.error(f"Error extracting text from page {page_num+1}: {e}")
                
                # Check memory usage after each batch
                self._check_memory_usage()
        
        return "\n".join(text_parts)
    
    def extract_text_with_pdfplumber(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pdfplumber with memory management.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Reading PDF with optimized pdfplumber: {pdf_path}")
        text_parts = []
        
        with self._open_pdfplumber(pdf_path) as pdf:
            # Get number of pages
            num_pages = len(pdf.pages)
            logger.info(f"PDF has {num_pages} pages")
            
            # Process pages in batches to manage memory
            batch_size = 3  # Process 3 pages at a time (pdfplumber uses more memory)
            for i in range(0, num_pages, batch_size):
                end_idx = min(i + batch_size, num_pages)
                logger.debug(f"Processing pages {i+1} to {end_idx} of {num_pages}")
                
                # Extract text from each page in batch
                for page_num in range(i, end_idx):
                    try:
                        page = pdf.pages[page_num]
                        page_text = page.extract_text(dedupe_chars=True) or ""
                        text_parts.append(page_text)
                    except Exception as e:
                        logger.error(f"Error extracting text from page {page_num+1}: {e}")
                
                # Check memory usage after each batch
                self._check_memory_usage()
        
        return "\n".join(text_parts)
    
    def extract_text(self, pdf_path: str, prefer_library: str = "pymupdf") -> Tuple[str, str]:
        """
        Extract text from PDF using the best available method.
        
        Args:
            pdf_path: Path to the PDF file
            prefer_library: Preferred library - "pymupdf" or "pdfplumber"
            
        Returns:
            Tuple of (extracted_text, library_used)
        """
        # First try the preferred library
        if prefer_library == "pymupdf":
            try:
                text = self.extract_text_with_pymupdf(pdf_path)
                if text.strip():
                    return text, "pymupdf"
                # If no text extracted, fall back to pdfplumber
                logger.info(f"PyMuPDF extracted no text, trying pdfplumber")
                text = self.extract_text_with_pdfplumber(pdf_path)
                return text, "pdfplumber"
            except Exception as e:
                logger.error(f"Error with PyMuPDF: {e}, falling back to pdfplumber")
                try:
                    text = self.extract_text_with_pdfplumber(pdf_path)
                    return text, "pdfplumber"
                except Exception as e2:
                    logger.error(f"Error with pdfplumber: {e2}")
                    return "", "none"
        else:
            try:
                text = self.extract_text_with_pdfplumber(pdf_path)
                if text.strip():
                    return text, "pdfplumber"
                # If no text extracted, fall back to PyMuPDF
                logger.info(f"pdfplumber extracted no text, trying PyMuPDF")
                text = self.extract_text_with_pymupdf(pdf_path)
                return text, "pymupdf"
            except Exception as e:
                logger.error(f"Error with pdfplumber: {e}, falling back to PyMuPDF")
                try:
                    text = self.extract_text_with_pymupdf(pdf_path)
                    return text, "pymupdf"
                except Exception as e2:
                    logger.error(f"Error with PyMuPDF: {e2}")
                    return "", "none"
                    
    def extract_images(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Extract images from PDF with memory management.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of paths to extracted images
        """
        logger.info(f"Extracting images from PDF: {pdf_path}")
        image_paths = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        with self._open_pymupdf(pdf_path) as pdf:
            # Get number of pages
            num_pages = len(pdf)
            
            # Process pages in batches
            batch_size = 3
            for i in range(0, num_pages, batch_size):
                end_idx = min(i + batch_size, num_pages)
                logger.debug(f"Extracting images from pages {i+1} to {end_idx}")
                
                # Extract images from each page in batch
                for page_num in range(i, end_idx):
                    try:
                        page = pdf[page_num]
                        image_list = page.get_images(full=True)
                        
                        # Process images in this page
                        for img_idx, img_info in enumerate(image_list):
                            xref = img_info[0]
                            try:
                                base_image = pdf.extract_image(xref)
                                image_bytes = base_image["image"]
                                image_ext = base_image["ext"]
                                
                                # Save the image
                                img_filename = f"page{page_num+1}_img{img_idx+1}.{image_ext}"
                                img_path = os.path.join(output_dir, img_filename)
                                with open(img_path, "wb") as img_file:
                                    img_file.write(image_bytes)
                                
                                image_paths.append(img_path)
                            except Exception as e:
                                logger.error(f"Error extracting image {img_idx} from page {page_num+1}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing page {page_num+1} for images: {e}")
                
                # Check memory usage after each batch
                self._check_memory_usage()
        
        return image_paths
