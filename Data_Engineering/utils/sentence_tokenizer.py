#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Sentence Tokenizer for French Transcripts

This module provides enhanced sentence tokenization for French transcripts,
particularly from sources like YouTube that may lack proper punctuation.
"""

import re
from typing import List, Dict, Any, Optional
import logging
import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

class ImprovedSentenceTokenizer:
    """
    Enhanced sentence tokenizer for French transcripts with missing punctuation.
    
    Uses a combination of rule-based detection and NLP to identify sentence boundaries
    in text with limited or missing punctuation.
    """
    
    def __init__(self, nlp_model: Language = None):
        """
        Initialize the sentence tokenizer.
        
        Args:
            nlp_model: Optional pre-loaded spaCy model
        """
        self.nlp = nlp_model
        
        # Patterns to identify potential sentence boundaries in unpunctuated text
        self.boundary_patterns = [
            # Common discourse markers that often start sentences
            r'\b(mais|donc|alors|ensuite|puis|enfin|cependant|toutefois|néanmoins|par contre|pourtant)\b',
            
            # Temporal markers - often start new thoughts
            r'\b(aujourd\'hui|maintenant|demain|hier|avant|après|pendant|durant|désormais|auparavant)\b',
            
            # Speaker changes
            r'\b[A-Z][a-zA-Z\s]*\s*:',
            
            # Question words at beginning of potential sentences
            r'\b(qui|que|quoi|quand|comment|pourquoi|où|combien)\b',
            
            # Common conjunctions that might indicate new sentences
            r'\b(car|parce que|puisque|ainsi|c\'est pourquoi)\b'
        ]
    
    def tokenize(self, text: str, aggressive: bool = False) -> List[str]:
        """
        Tokenize text into sentences with enhanced handling for unpunctuated transcripts.
        
        Args:
            text: Text to tokenize
            aggressive: Whether to use more aggressive tokenization for unpunctuated text
            
        Returns:
            List of sentence strings
        """
        # If text already has reasonable punctuation, use spaCy
        if self._has_sufficient_punctuation(text):
            logger.debug("Text has sufficient punctuation, using standard tokenization")
            return self._tokenize_with_spacy(text)
        
        # For unpunctuated text, use our enhanced approach
        logger.debug("Text lacks sufficient punctuation, using enhanced tokenization")
        return self._tokenize_unpunctuated(text, aggressive)
    
    def _has_sufficient_punctuation(self, text: str) -> bool:
        """
        Check if the text has enough punctuation to use standard tokenization.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating whether text has sufficient punctuation
        """
        # Count sentence-ending punctuation
        endings = re.findall(r'[.?!]', text)
        text_length = len(text)
        
        # Get rough estimate of expected sentences (1 per ~100 chars is reasonable)
        expected_sentences = max(1, text_length / 100)
        
        # If we have at least 60% of expected sentence endings, consider it sufficient
        return len(endings) >= (0.6 * expected_sentences)
    
    def _tokenize_with_spacy(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy's sentence boundary detection.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of sentence strings
        """
        if not self.nlp:
            raise ValueError("No spaCy model provided. Set nlp_model when initializing.")
            
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def _tokenize_unpunctuated(self, text: str, aggressive: bool = False) -> List[str]:
        """
        Tokenize text with missing or limited punctuation.
        
        Args:
            text: Text to tokenize
            aggressive: Whether to use more aggressive tokenization
            
        Returns:
            List of sentence strings
        """
        # First, try to add periods where they likely belong
        text = self._add_implied_periods(text)
        
        # Split on actual periods, question marks, and exclamation marks
        segments = re.split(r'(?<=[.!?])\s+', text)
        
        # Further split unpunctuated long segments
        sentences = []
        for segment in segments:
            if len(segment) < 150 and not aggressive:  # Short segments are likely already sentences
                sentences.append(segment)
                continue
                
            # For longer segments without punctuation, try to find natural breaks
            potential_boundaries = self._find_boundary_candidates(segment)
            if not potential_boundaries:
                sentences.append(segment)
                continue
                
            # Split the segment at the identified boundaries
            last_pos = 0
            for pos in sorted(potential_boundaries):
                if pos - last_pos > 30:  # Ensure minimum length for segments
                    sentences.append(segment[last_pos:pos].strip())
                    last_pos = pos
            
            # Add the last segment
            if last_pos < len(segment):
                sentences.append(segment[last_pos:].strip())
        
        # Clean up and return
        return [s.strip() for s in sentences if s.strip()]
    
    def _add_implied_periods(self, text: str) -> str:
        """
        Add periods where they seem to be missing.
        
        Args:
            text: Text to process
            
        Returns:
            Text with added periods
        """
        # Add periods after speaker identifications
        text = re.sub(r'([A-Z][a-z]+\s*:)([A-Z])', r'\1 \2', text)
        
        # Add periods before capital letters that follow lowercase letter if not already punctuated
        text = re.sub(r'(\s)([a-zàáâäæçèéêëìíîïñòóôöùúûüÿ])(\s+)([A-ZÀÁÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÖÙÚÛÜŸ])', r'\1\2. \4', text)
        
        return text
    
    def _find_boundary_candidates(self, text: str) -> List[int]:
        """
        Find potential sentence boundaries in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of character positions where sentences might end
        """
        boundaries = []
        
        # Find matches for each boundary pattern
        for pattern in self.boundary_patterns:
            for match in re.finditer(pattern, text):
                start = match.start()
                
                # Ensure we're not in the middle of a word (check for space before)
                if start > 0 and text[start-1].isspace():
                    boundaries.append(start)
        
        # Add boundaries at capital letters after spaces if they follow a lowercase letter
        for match in re.finditer(r'([a-z])\s+([A-Z])', text):
            boundaries.append(match.start(2))
            
        return boundaries
