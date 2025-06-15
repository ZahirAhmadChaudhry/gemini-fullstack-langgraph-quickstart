"""
Temporal context distinction module for French sustainability discourse.

This module implements rule-based detection of verb tenses and temporal markers
to distinguish between present (2023) and future (2050) contexts.
"""

import os
import csv
import re
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
import spacy
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemporalContextAnalyzer:
    """
    Analyzes temporal context in French text to distinguish present vs. future references.
    """
    
    def __init__(self, 
                 present_markers_path: Optional[str] = None,
                 future_markers_path: Optional[str] = None,
                 verb_tense_patterns_path: Optional[str] = None,
                 spacy_model: str = "fr_core_news_lg"):
        """
        Initialize the temporal context analyzer.
        
        Args:
            present_markers_path: Path to file with present temporal markers
            future_markers_path: Path to file with future temporal markers
            verb_tense_patterns_path: Path to file with verb tense patterns
            spacy_model: French spaCy model to use
        """
        self.present_markers = set()
        self.future_markers = set()
        self.verb_tense_patterns = {}
        self.nlp = None
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise
            
        # Load temporal markers if provided
        if present_markers_path and os.path.exists(present_markers_path):
            self.load_temporal_markers(present_markers_path, "present")
        
        if future_markers_path and os.path.exists(future_markers_path):
            self.load_temporal_markers(future_markers_path, "future")
        
        # Load verb tense patterns if provided
        if verb_tense_patterns_path and os.path.exists(verb_tense_patterns_path):
            self.load_verb_tense_patterns(verb_tense_patterns_path)
        
        # Initialize French verb tense detection rules
        self.initialize_default_rules()
    
    def initialize_default_rules(self):
        """
        Initialize default rules for French verb tense detection if no patterns file provided.
        """
        # Default patterns if none provided
        if not self.verb_tense_patterns:
            # Format: (pattern, is_regex)
            self.verb_tense_patterns = {
                # Futur Simple patterns
                "futur_simple": [
                    (r'\b\w+rai\b', True),    # je parlerai
                    (r'\b\w+ras\b', True),    # tu parleras
                    (r'\b\w+ra\b', True),     # il/elle parlera
                    (r'\b\w+rons\b', True),   # nous parlerons
                    (r'\b\w+rez\b', True),    # vous parlerez
                    (r'\b\w+ront\b', True),   # ils/elles parleront
                ],
                # Futur Proche patterns
                "futur_proche": [
                    (r'\bvais\s+\w+er\b', True),      # je vais parler
                    (r'\bvas\s+\w+er\b', True),       # tu vas parler
                    (r'\bva\s+\w+er\b', True),        # il/elle va parler
                    (r'\ballons\s+\w+er\b', True),    # nous allons parler
                    (r'\ballez\s+\w+er\b', True),     # vous allez parler
                    (r'\bvont\s+\w+er\b', True),      # ils/elles vont parler
                ],
                # Conditionnel patterns
                "conditionnel": [
                    (r'\b\w+rais\b', True),    # je parlerais
                    (r'\b\w+rais\b', True),    # tu parlerais
                    (r'\b\w+rait\b', True),    # il/elle parlerait
                    (r'\b\w+rions\b', True),   # nous parlerions
                    (r'\b\w+riez\b', True),    # vous parleriez
                    (r'\b\w+raient\b', True),  # ils/elles parleraient
                ]
            }
            
        # Default temporal markers if none provided
        if not self.present_markers:
            self.present_markers = {
                "aujourd'hui", "maintenant", "actuellement", "en ce moment", 
                "présentement", "à l'heure actuelle", "de nos jours", 
                "cette année", "cette semaine", "ce mois-ci", "cette époque",
                "2023"  # Specific to this project
            }
            
        if not self.future_markers:
            self.future_markers = {
                "demain", "bientôt", "prochainement", "à l'avenir", "plus tard",
                "dans le futur", "à terme", "dans quelques années", 
                "au cours des prochaines années", "d'ici là", "d'ici peu",
                "dans dix ans", "dans vingt ans", "dans trente ans",
                "2050"  # Specific to this project
            }
    
    def load_temporal_markers(self, filepath: str, marker_type: str) -> None:
        """
        Load temporal markers from file.
        
        Args:
            filepath: Path to temporal markers file
            marker_type: Type of markers ('present' or 'future')
        """
        marker_set = self.present_markers if marker_type == "present" else self.future_markers
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.csv'):
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # Skip empty rows
                            marker_set.add(row[0].lower().strip())
                else:
                    for line in f:
                        marker = line.strip().lower()
                        if marker:  # Skip empty lines
                            marker_set.add(marker)
            
            logger.info(f"Loaded {len(marker_set)} {marker_type} temporal markers")
        except Exception as e:
            logger.error(f"Error loading {marker_type} temporal markers: {e}")
    
    def load_verb_tense_patterns(self, filepath: str) -> None:
        """
        Load verb tense patterns from file.
        
        Args:
            filepath: Path to verb tense patterns file
        """
        try:
            patterns = defaultdict(list)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.csv'):
                    reader = csv.reader(f)
                    header = next(reader, None)
                    
                    # Expect format: tense,pattern,is_regex
                    for row in reader:
                        if len(row) >= 3:
                            tense = row[0].strip()
                            pattern = row[1].strip()
                            is_regex = row[2].lower() in ('true', '1', 'yes', 'y')
                            patterns[tense].append((pattern, is_regex))
                else:
                    # Simple format: tense<tab>pattern<tab>is_regex
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            tense = parts[0].strip()
                            pattern = parts[1].strip()
                            is_regex = parts[2].lower() in ('true', '1', 'yes', 'y')
                            patterns[tense].append((pattern, is_regex))
            
            self.verb_tense_patterns = dict(patterns)
            logger.info(f"Loaded verb tense patterns for {len(self.verb_tense_patterns)} tenses")
        except Exception as e:
            logger.error(f"Error loading verb tense patterns: {e}")
    
    def detect_temporal_markers(self, text: str) -> Dict[str, List[str]]:
        """
        Detect temporal markers in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with present and future markers found
        """
        text_lower = text.lower()
        
        present_found = []
        future_found = []
        
        # Check for present markers
        for marker in self.present_markers:
            if marker in text_lower:
                present_found.append(marker)
        
        # Check for future markers
        for marker in self.future_markers:
            if marker in text_lower:
                future_found.append(marker)
        
        return {
            "present": present_found,
            "future": future_found
        }
    
    def detect_verb_tenses(self, doc: spacy.tokens.Doc) -> Dict[str, int]:
        """
        Detect verb tenses in a spaCy document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Dictionary with counts of different tenses
        """
        text = doc.text.lower()
        tense_counts = Counter()
        
        # Detect tenses using patterns
        for tense, patterns in self.verb_tense_patterns.items():
            for pattern, is_regex in patterns:
                if is_regex:
                    matches = re.finditer(pattern, text)
                    for _ in matches:
                        tense_counts[tense] += 1
                else:
                    count = text.count(pattern)
                    tense_counts[tense] += count
        
        # Use spaCy morphological analysis for Présent tense (more challenging)
        for token in doc:
            if token.pos_ == "VERB" or token.pos_ == "AUX":
                morph = token.morph.get("Tense")
                if morph and morph[0] == "Pres":
                    # Present tense detected via morphology
                    tense_counts["present"] += 1
        
        return dict(tense_counts)
    
    def disambiguate_present_tense(self, doc: spacy.tokens.Doc, tense_counts: Dict[str, int]) -> Dict[str, int]:
        """
        Disambiguate Présent tense usage for future reference.
        
        In French, present tense is often used to refer to near future.
        This function attempts to disambiguate based on context.
        
        Args:
            doc: spaCy Doc object
            tense_counts: Initial tense counts
            
        Returns:
            Updated tense counts with disambiguated present tense
        """
        # Copy the counts to avoid modifying the original
        updated_counts = tense_counts.copy()
        
        # Present tense used for future reference typically has future temporal markers
        text_lower = doc.text.lower()
        
        # Check for future temporal markers that often signal present-for-future usage
        future_signals = [
            "demain", "bientôt", "prochainement", "à l'avenir", "plus tard",
            "la semaine prochaine", "le mois prochain", "l'année prochaine", 
            "dans", "d'ici"
        ]
        
        future_signal_count = sum(1 for signal in future_signals if signal in text_lower)
        present_count = updated_counts.get("present", 0)
        
        # If we have present tense verbs and future signals,
        # reassign some present tense verbs to "present_for_future"
        if present_count > 0 and future_signal_count > 0:
            # For a single present tense verb with immediate future signal, always mark it as present_for_future
            if present_count == 1 and any(signal in text_lower for signal in ["demain", "bientôt", "prochainement"]):
                present_for_future_count = 1
            else:
                # The more future signals, the more present tense verbs likely refer to future
                # But cap at 80% of present tense verbs
                present_for_future_count = min(
                    max(1, int(present_count * 0.4 * future_signal_count)), 
                    int(present_count * 0.8) or 1  # Ensure at least 1 if calculation would be 0
                )
            
            if present_for_future_count > 0:
                updated_counts["present"] -= present_for_future_count
                updated_counts["present_for_future"] = present_for_future_count
                
        return updated_counts
    
    def analyze_temporal_context(self, text: str) -> Dict[str, Any]:
        """
        Analyze temporal context of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with temporal context analysis
        """
        if not text:
            return {
                "context": "unknown",
                "confidence": 0.0,
                "evidence": {
                    "markers": {"present": [], "future": []},
                    "tenses": {}
                }
            }
            
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Detect temporal markers
        markers = self.detect_temporal_markers(text)
        
        # Detect verb tenses
        tense_counts = self.detect_verb_tenses(doc)
        
        # Disambiguate present tense usage
        tense_counts = self.disambiguate_present_tense(doc, tense_counts)
        
        # Calculate evidence scores
        present_evidence = 0.0
        future_evidence = 0.0
        
        # 1. Temporal markers evidence
        present_marker_count = len(markers["present"])
        future_marker_count = len(markers["future"])
        
        # Direct year references have highest weight
        present_year_ref = 1 if "2023" in [m.strip() for m in markers["present"]] else 0
        future_year_ref = 1 if "2050" in [m.strip() for m in markers["future"]] else 0
        
        present_evidence += 0.5 * present_marker_count + 2.0 * present_year_ref
        future_evidence += 0.5 * future_marker_count + 2.0 * future_year_ref
        
        # 2. Verb tense evidence
        # Present tense (when not used for future)
        present_tense_count = tense_counts.get("present", 0)
        present_evidence += 0.3 * present_tense_count
        
        # Future tenses
        futur_simple_count = tense_counts.get("futur_simple", 0)
        futur_proche_count = tense_counts.get("futur_proche", 0)
        conditionnel_count = tense_counts.get("conditionnel", 0)
        present_for_future_count = tense_counts.get("present_for_future", 0)
        
        future_evidence += 0.7 * futur_simple_count
        future_evidence += 0.7 * futur_proche_count
        future_evidence += 0.4 * conditionnel_count
        future_evidence += 0.5 * present_for_future_count
        
        # 3. Determine the predominant context
        if future_evidence > present_evidence:
            context = "future"
            confidence = min(0.5 + 0.1 * future_evidence, 0.95)
        elif present_evidence > future_evidence:
            context = "present"
            confidence = min(0.5 + 0.1 * present_evidence, 0.95)
        else:
            # If equal evidence, default to present
            context = "unknown"
            confidence = 0.5
            
        return {
            "context": context,
            "confidence": float(confidence),
            "evidence": {
                "markers": markers,
                "tenses": tense_counts
            }
        }
    
    def process_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single text segment for temporal context distinction.
        
        Args:
            segment: Text segment to analyze
            
        Returns:
            Segment with added temporal context metadata
        """
        # Extract text from segment
        text = segment.get("text", "")
        if isinstance(text, list):
            # Join if text is a list of sentences
            text = " ".join(text)
            
        if not text:
            logger.warning(f"Empty text in segment {segment.get('segment_id', 'unknown')}")
            segment["temporal_context"] = {
                "context": "unknown", 
                "confidence": 0.0
            }
            return segment
            
        # Analyze temporal context
        temporal_context = self.analyze_temporal_context(text)
        
        # Add temporal context to segment
        segment["temporal_context"] = temporal_context
        
        return segment
    
    def process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple text segments for temporal context distinction.
        
        Args:
            segments: List of text segments to analyze
            
        Returns:
            Segments with added temporal context metadata
        """
        if not segments:
            return []
            
        # Process each segment
        processed_segments = []
        for segment in segments:
            processed = self.process_segment(segment)
            processed_segments.append(processed)
            
        logger.info(f"Processed {len(processed_segments)} segments for temporal context distinction")
        return processed_segments

