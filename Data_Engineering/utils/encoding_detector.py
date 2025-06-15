#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Encoding Detector for French Text

This module provides enhanced encoding detection for French language texts,
with special focus on handling diacritics and mojibake patterns.
"""

import re
import chardet
import ftfy
from typing import Tuple, Dict, List, Optional, Any
import logging
from pathlib import Path
import codecs

logger = logging.getLogger(__name__)

class RobustEncodingDetector:
    """
    Enhanced encoding detector for French text.
    
    Implements a multi-step approach to reliably detect text encoding with
    special handling for French language patterns and diacritics.
    """
    
    def __init__(self):
        """Initialize the encoding detector."""
        # French-specific patterns to validate proper encoding
        self.french_patterns = [
            # Common French diacritics
            r'[àáâäæçèéêëìíîïñòóôöùúûüÿÀÁÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÖÙÚÛÜŸ]',
            # Common French words with accents
            r'\b(être|où|à|français|voilà|déjà|très|après|première)\b',
            # Common French punctuation patterns
            r' [?!] | [«»] | \b[A-Z][a-z]*\s*: ',
        ]
        
        # Mojibake patterns to detect encoding errors
        self.mojibake_patterns = [
            # Common mojibake sequences for UTF-8 interpreted as ISO-8859-1/Windows-1252
            r'Ã©|Ã¨|Ãª|Ã´|Ã®|Ã»|Ã§',
            # Common mojibake sequences for ISO-8859-1 interpreted as UTF-8
            r'é(?:e|a)|è(?:e|a)|ê(?:e|a)|ô(?:a|e)|î(?:e|a)|ù(?:e|a)|ç(?:a|e)',
            # Gibberish character sequences unlikely in proper French
            r'[Ã±Ã¿Ã²Ã¡]{2,}'
        ]
        
        # Encoding candidates to try in order of preference
        self.encoding_candidates = [
            'utf-8',
            'iso-8859-1',
            'windows-1252',
            'latin-1',
            'utf-16',
            'cp1252'
        ]
        
    def detect_encoding(self, file_path: Path) -> Tuple[str, float]:
        """
        Detect the encoding of a text file with enhanced accuracy for French.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Tuple of (detected_encoding, confidence_score)
        """
        logger.info(f"Detecting encoding for {file_path}")
        
        # Read raw bytes from file
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
            
        # 1. Try UTF-8 first (most reliable standard)
        try:
            raw_bytes.decode('utf-8')
            logger.info(f"File {file_path} is valid UTF-8")
            return 'utf-8', 1.0
        except UnicodeDecodeError:
            logger.debug(f"File {file_path} is not valid UTF-8, trying other encodings")
        
        # 2. Use chardet library for initial guess
        chardet_result = chardet.detect(raw_bytes)
        detected_encoding = chardet_result['encoding'] or 'utf-8'  
        confidence = chardet_result['confidence']
        logger.debug(f"Chardet detection: {detected_encoding} with confidence {confidence}")
        
        # 3. Cross-validate with multiple detection strategies
        encodings_to_try = [detected_encoding] + [enc for enc in self.encoding_candidates if enc != detected_encoding]
        best_encoding = None
        best_score = -1
        
        for encoding in encodings_to_try:
            try:
                decoded_text = raw_bytes.decode(encoding, errors='replace')
                
                # Score the decoded text based on French patterns
                french_score = self._score_french_text(decoded_text)
                mojibake_score = self._detect_mojibake(decoded_text)
                
                # Calculate combined score
                # Higher french_score is better, lower mojibake_score is better
                combined_score = french_score * (1 - mojibake_score)
                
                logger.debug(f"Encoding {encoding}: French score={french_score:.2f}, "
                           f"Mojibake score={mojibake_score:.2f}, Combined={combined_score:.2f}")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_encoding = encoding
            except Exception as e:
                logger.debug(f"Error with encoding {encoding}: {e}")
        
        if best_encoding:
            logger.info(f"Selected best encoding for {file_path}: {best_encoding} with score {best_score:.2f}")
            return best_encoding, best_score
        else:
            logger.warning(f"Falling back to default UTF-8 for {file_path}")
            return 'utf-8', 0.5
    
    def _score_french_text(self, text: str) -> float:
        """
        Score text based on the presence of French language patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Score between 0.0 and 1.0 (higher is more likely French)
        """
        total_patterns = len(self.french_patterns)
        matches = 0
        
        for pattern in self.french_patterns:
            if re.search(pattern, text):
                matches += 1
                
        # Return normalized score
        return matches / total_patterns if total_patterns > 0 else 0.5
    
    def _detect_mojibake(self, text: str) -> float:
        """
        Detect mojibake patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Score between 0.0 and 1.0 (higher means more mojibake detected)
        """
        # Count total mojibake occurrences
        total_mojibake = 0
        for pattern in self.mojibake_patterns:
            matches = re.findall(pattern, text)
            total_mojibake += len(matches)
        
        # Normalize by text length to get proportion
        text_length = len(text)
        if text_length == 0:
            return 0.0
        
        mojibake_ratio = min(1.0, total_mojibake / (text_length * 0.1))  # Cap at 1.0
        return mojibake_ratio
    
    def fix_mojibake(self, text: str) -> str:
        """
        Fix common mojibake patterns in French text.
        
        Args:
            text: Text with potential mojibake
            
        Returns:
            Corrected text
        """
        # Use ftfy to fix text encoding issues
        fixed_text = ftfy.fix_text(text)
        
        # Additional French-specific fixes
        # Replace common mojibake patterns
        fixes = [
            (r'Ã©', 'é'),
            (r'Ã¨', 'è'),
            (r'Ãª', 'ê'),
            (r'Ã´', 'ô'),
            (r'Ã®', 'î'),
            (r'Ã»', 'û'),
            (r'Ã§', 'ç'),
            (r'Ã ', 'à'),
            # Add more patterns as needed
        ]
        
        for pattern, replacement in fixes:
            fixed_text = re.sub(pattern, replacement, fixed_text)
        
        return fixed_text
    
    def decode_and_fix(self, file_path: Path) -> str:
        """
        Detect encoding, decode the file, and fix any mojibake issues.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Properly decoded and fixed text
        """
        encoding, _ = self.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()
                
            # Fix any remaining mojibake issues
            fixed_text = self.fix_mojibake(text)
            
            return fixed_text
        except Exception as e:
            logger.error(f"Error decoding {file_path} with encoding {encoding}: {e}")
            # Fall back to UTF-8 with error replacement
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
