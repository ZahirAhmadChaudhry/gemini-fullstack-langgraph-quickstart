"""
Paradox detection module using rule-based approaches for French sustainability discourse.

This module implements rules for detecting paradoxes based on linguistic patterns,
including lexical cues, syntactic patterns, and discourse markers.
"""

import os
import csv
import re
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
import spacy
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParadoxDetector:
    """
    Detects paradoxes in French sustainability discourse using rule-based methods.
    """
    
    def __init__(self, 
                 sustainability_paradox_terms_path: Optional[str] = None,
                 antonyms_path: Optional[str] = None, 
                 tension_keywords_path: Optional[str] = None,
                 confidence_threshold: float = 0.6,
                 spacy_model: str = "fr_core_news_lg"):
        """
        Initialize the paradox detector.
        
        Args:
            sustainability_paradox_terms_path: Path to sustainability paradox terms file
            antonyms_path: Path to French antonyms file
            tension_keywords_path: Path to tension keywords file
            confidence_threshold: Threshold for classifying a paradox
            spacy_model: French spaCy model to use
        """
        self.confidence_threshold = confidence_threshold
        self.sustainability_terms = set()
        self.antonyms = defaultdict(set)
        self.tension_keywords = set()
        self.nlp = None
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise
            
        # Load sustainability paradox terms if provided
        if sustainability_paradox_terms_path and os.path.exists(sustainability_paradox_terms_path):
            self.load_sustainability_paradox_terms(sustainability_paradox_terms_path)
        
        # Load antonyms if provided
        if antonyms_path and os.path.exists(antonyms_path):
            self.load_antonyms(antonyms_path)
        
        # Load tension keywords if provided
        if tension_keywords_path and os.path.exists(tension_keywords_path):
            self.load_tension_keywords(tension_keywords_path)
    
    def load_sustainability_paradox_terms(self, filepath: str) -> None:
        """
        Load sustainability domain paradox terms from file.
        
        Args:
            filepath: Path to sustainability paradox terms file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.csv'):
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # Skip empty rows
                            self.sustainability_terms.add(row[0].lower().strip())
                else:
                    for line in f:
                        term = line.strip().lower()
                        if term:  # Skip empty lines
                            self.sustainability_terms.add(term)
            
            logger.info(f"Loaded {len(self.sustainability_terms)} sustainability paradox terms")
        except Exception as e:
            logger.error(f"Error loading sustainability paradox terms: {e}")
    
    def load_antonyms(self, filepath: str) -> None:
        """
        Load French antonyms from file.
        
        Args:
            filepath: Path to antonyms file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.csv'):
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            word1 = row[0].lower().strip()
                            word2 = row[1].lower().strip()
                            self.antonyms[word1].add(word2)
                            self.antonyms[word2].add(word1)
                else:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            word1 = parts[0].lower()
                            word2 = parts[1].lower()
                            self.antonyms[word1].add(word2)
                            self.antonyms[word2].add(word1)
            
            logger.info(f"Loaded antonyms for {len(self.antonyms)} words")
        except Exception as e:
            logger.error(f"Error loading antonyms: {e}")
    
    def load_tension_keywords(self, filepath: str) -> None:
        """
        Load tension keywords from file.
        
        Args:
            filepath: Path to tension keywords file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.csv'):
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # Skip empty rows
                            self.tension_keywords.add(row[0].lower().strip())
                else:
                    for line in f:
                        term = line.strip().lower()
                        if term:  # Skip empty lines
                            self.tension_keywords.add(term)
            
            logger.info(f"Loaded {len(self.tension_keywords)} tension keywords")
        except Exception as e:
            logger.error(f"Error loading tension keywords: {e}")
    
    def detect_antonym_pairs(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """
        Detect antonym pairs in text.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of antonym pair detections
        """
        if not self.antonyms:
            return []
            
        detections = []
        tokens = [token.lemma_.lower() for token in doc]
        
        # Check for antonym pairs within a window of 10 tokens
        window_size = 10
        for i, token in enumerate(tokens):
            if token in self.antonyms:
                # Check for antonyms within window
                start_idx = max(0, i - window_size)
                end_idx = min(len(tokens), i + window_size)
                window = tokens[start_idx:end_idx]
                
                for j, other_token in enumerate(window):
                    if other_token in self.antonyms[token]:
                        abs_j = start_idx + j
                        distance = abs(i - abs_j)
                        
                        if distance > 0:  # Avoid self-matches
                            confidence = 0.9 * (1.0 - distance / window_size)
                            
                            if confidence >= self.confidence_threshold:
                                detection = {
                                    "rule": "antonym_pair",
                                    "word1": token,
                                    "word2": other_token,
                                    "positions": (i, abs_j),
                                    "confidence": float(confidence)
                                }
                                detections.append(detection)
        
        return detections
    
    def detect_negated_repetition(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """
        Detect words that appear in both positive and negative form.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of negated repetition detections
        """
        detections = []
        
        # Find negation cues
        negation_tokens = []
        for i, token in enumerate(doc):
            if token.dep_ == "neg" or token.text.lower() in {"ne", "n'", "pas", "jamais", "aucun", "aucune"}:
                negation_tokens.append((i, token))
        
        # For each content word, check if it appears in both positive and negated form
        content_words = defaultdict(list)
        for i, token in enumerate(doc):
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop:
                content_words[token.lemma_.lower()].append(i)
        
        for word, positions in content_words.items():
            if len(positions) < 2:
                continue
                
            # Check if at least one occurrence is negated
            has_positive = False
            has_negative = False
            
            for pos in positions:
                # Check if word is in scope of a negation
                is_negated = False
                for neg_pos, _ in negation_tokens:
                    # Check if negation is close to word and precedes it
                    if 0 < (pos - neg_pos) < 5:  # Typical French negation scope
                        is_negated = True
                        break
                
                if is_negated:
                    has_negative = True
                else:
                    has_positive = True
            
            # If word appears in both positive and negative form
            if has_positive and has_negative:
                confidence = 0.8
                detection = {
                    "rule": "negated_repetition",
                    "word": word,
                    "positions": positions,
                    "confidence": float(confidence)
                }
                detections.append(detection)
        
        return detections
    
    def detect_sustainability_tensions(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """
        Detect sustainability domain tensions and common paradox themes.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of sustainability tension detections
        """
        if not self.sustainability_terms or not self.tension_keywords:
            return []
            
        detections = []
        text_lower = doc.text.lower()
        
        # Check for sustainability terms
        found_sustainability_terms = []
        for term in self.sustainability_terms:
            if term in text_lower:
                found_sustainability_terms.append(term)
        
        if not found_sustainability_terms:
            return []
            
        # Check for tension keywords
        found_tension_keywords = []
        for keyword in self.tension_keywords:
            if keyword in text_lower:
                found_tension_keywords.append(keyword)
        
        # If both sustainability term and tension keyword present
        if found_sustainability_terms and found_tension_keywords:
            # Base confidence level starts at 0.5
            confidence = 0.5
            
            # Add more confidence based on number of tension keywords (up to 0.2)
            confidence += min(0.1 * len(found_tension_keywords), 0.2)
            
            # Check for explicit connection between sustainability terms and tension keywords
            # by looking at their proximity in the text
            max_proximity_score = 0
            for term in found_sustainability_terms:
                term_index = text_lower.find(term)
                if term_index >= 0:
                    for keyword in found_tension_keywords:
                        keyword_index = text_lower.find(keyword)
                        if keyword_index >= 0:
                            # Calculate distance between term and keyword
                            distance = abs(term_index - keyword_index)
                            # Shorter distance means stronger connection
                            proximity_score = max(0, 0.2 - (distance / 500) * 0.2)
                            max_proximity_score = max(max_proximity_score, proximity_score)
            
            # Add proximity score to confidence
            confidence += max_proximity_score
            
            # Check for explicit tension signal phrases
            explicit_tension_signals = [
                "entre", "opposé", "contraire", "conflit", "contradiction",
                "tension entre", "paradoxe entre", "dilemme entre"
            ]
            
            # Lower confidence slightly for borderline cases that just mention "compromis"
            # without strong paradoxical context
            if "compromis" in found_tension_keywords and len(found_tension_keywords) == 1:
                # Check if this is a simple mention without strong paradoxical framing
                if not any(signal in text_lower for signal in ["conflit", "contradiction", "tension", "paradoxe", "dilemme"]):
                    confidence -= 0.1
            
            # Boost confidence for explicit tension signals
            for signal in explicit_tension_signals:
                if signal in text_lower:
                    confidence += 0.1
                    break
            
            # Cap confidence at 0.9
            confidence = min(confidence, 0.9)
            
            if confidence >= self.confidence_threshold:
                detection = {
                    "rule": "sustainability_tension",
                    "sustainability_terms": found_sustainability_terms,
                    "tension_keywords": found_tension_keywords,
                    "confidence": float(confidence)
                }
                detections.append(detection)
        
        return detections
    
    def detect_contrastive_structures(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """
        Detect contrastive discourse structures that signal paradoxes.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of contrastive structure detections
        """
        detections = []
        
        # Common French contrastive markers
        contrastive_patterns = [
            r'\bd\'une part\b.*\bd\'autre part\b',
            r'\bd\'un côté\b.*\bde l\'autre\b',
            r'\bnon seulement\b.*\bmais aussi\b',
            r'\bà la fois\b.*\bet\b',
            r'\btout en\b',
            r'\bmais\b',
            r'\bcependant\b',
            r'\btoutefois\b',
            r'\bnéanmoins\b',
            r'\bpourtant\b',
            r'\ben revanche\b',
            r'\bau contraire\b',
            r'\bbien que\b',
            r'\bmalgré\b',
            r'\ben dépit de\b',
            r'\bmême si\b',
        ]
        
        text = doc.text.lower()
        
        # Check for contrastive patterns using regex
        for pattern in contrastive_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                span = match.span()
                # For simple markers, check that surrounding text has diverse sentiment
                if len(match.group()) < 15:  # Short patterns like "mais", "cependant"
                    # Extract context before and after marker
                    context_before = text[:span[0]].strip()
                    context_after = text[span[1]:].strip()
                    
                    # Skip if either context is too short
                    if len(context_before) < 10 or len(context_after) < 10:
                        continue
                    
                    confidence = 0.65
                else:
                    # For longer patterns (e.g., "d'une part... d'autre part"), the structure itself
                    # is a stronger indicator of potential paradox
                    confidence = 0.75
                
                if confidence >= self.confidence_threshold:
                    detection = {
                        "rule": "contrastive_structure",
                        "pattern": match.group(),
                        "span": span,
                        "confidence": float(confidence)
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_paradoxes(self, text: str) -> Dict[str, Any]:
        """
        Detect paradoxes in text using multiple rules.
        
        Args:
            text: Text segment to analyze
            
        Returns:
            Dictionary with paradox detection results
        """
        if not text:
            return {
                "is_paradox": False,
                "confidence": 0.0,
                "detections": []
            }
            
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Apply detection rules
        all_detections = []
        
        # 1. Antonym pairs
        antonym_detections = self.detect_antonym_pairs(doc)
        all_detections.extend(antonym_detections)
        
        # 2. Negated repetition
        negation_detections = self.detect_negated_repetition(doc)
        all_detections.extend(negation_detections)
        
        # 3. Sustainability tensions
        tension_detections = self.detect_sustainability_tensions(doc)
        all_detections.extend(tension_detections)
        
        # 4. Contrastive structures
        contrastive_detections = self.detect_contrastive_structures(doc)
        all_detections.extend(contrastive_detections)
        
        # Calculate overall confidence
        if all_detections:
            # Weight detections by their confidence and rule type
            weighted_sum = 0.0
            total_weight = 0.0
            
            for detection in all_detections:
                rule = detection["rule"]
                confidence = detection["confidence"]
                
                # Assign weights to different rule types
                if rule == "antonym_pair":
                    weight = 1.2
                elif rule == "negated_repetition":
                    weight = 1.0
                elif rule == "sustainability_tension":
                    weight = 0.9
                elif rule == "contrastive_structure":
                    weight = 0.7
                else:
                    weight = 1.0
                    
                weighted_sum += confidence * weight
                total_weight += weight
            
            overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
            is_paradox = overall_confidence >= self.confidence_threshold
        else:
            overall_confidence = 0.0
            is_paradox = False
            
        return {
            "is_paradox": is_paradox,
            "confidence": float(overall_confidence),
            "detections": all_detections
        }
    
    def process_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single text segment for paradox detection.
        
        Args:
            segment: Text segment to analyze
            
        Returns:
            Segment with added paradox metadata
        """
        # Extract text from segment
        text = segment.get("text", "")
        if isinstance(text, list):
            # Join if text is a list of sentences
            text = " ".join(text)
            
        if not text:
            logger.warning(f"Empty text in segment {segment.get('segment_id', 'unknown')}")
            segment["paradox"] = {"is_paradox": False, "confidence": 0.0, "detections": []}
            return segment
            
        # Detect paradoxes
        paradox_result = self.detect_paradoxes(text)
        
        # Add paradox detection to segment
        segment["paradox"] = paradox_result
        
        return segment
    
    def process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple text segments for paradox detection.
        
        Args:
            segments: List of text segments to analyze
            
        Returns:
            Segments with added paradox metadata
        """
        if not segments:
            return []
            
        # Process each segment
        processed_segments = []
        for segment in segments:
            processed = self.process_segment(segment)
            processed_segments.append(processed)
            
        logger.info(f"Processed {len(processed_segments)} segments for paradox detection")
        return processed_segments

