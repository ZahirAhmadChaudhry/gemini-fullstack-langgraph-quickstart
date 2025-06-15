"""
Feature Engineering module for extracting and enhancing features from French text segments.

This module leverages the rich features from the new JSON format and creates
additional statistical and linguistic features for ML models.
"""

import logging
import numpy as np
import re
from typing import List, Dict, Any, Optional, Set
from collections import Counter

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# Configure logging
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Advanced feature engineering for French sustainability text analysis.
    """
    
    def __init__(self, spacy_model: str = "fr_core_news_lg"):
        """
        Initialize the feature engineering system.

        Args:
            spacy_model: French spaCy model for linguistic analysis
        """
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Some features will be limited.")

        self.spacy_model = spacy_model
        self.nlp = None
        
        # Feature extraction patterns
        self.temporal_patterns = {
            "present": [
                r"\b(aujourd'hui|maintenant|actuellement|présentement|en ce moment)\b",
                r"\b(cette année|ce mois|cette semaine)\b",
                r"\b(2023|2024)\b"
            ],
            "future": [
                r"\b(demain|bientôt|prochainement|ultérieurement)\b",
                r"\b(l'année prochaine|le mois prochain|la semaine prochaine)\b",
                r"\b(2050|2030|2040)\b",
                r"\b(dans le futur|à l'avenir|plus tard)\b"
            ]
        }
        
        self.sustainability_indicators = {
            "environmental": [
                "environnement", "écologie", "climat", "carbone", "émissions",
                "pollution", "biodiversité", "recyclage", "énergie renouvelable"
            ],
            "social": [
                "social", "société", "communauté", "équité", "justice",
                "inclusion", "diversité", "bien-être", "santé"
            ],
            "economic": [
                "économie", "économique", "financier", "coût", "investissement",
                "rentabilité", "profit", "croissance", "développement"
            ]
        }
        
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize spaCy model."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Linguistic features will be limited.")
            return

        try:
            self.nlp = spacy.load(self.spacy_model)
            logger.info(f"Loaded spaCy model: {self.spacy_model}")
        except Exception as e:
            logger.warning(f"Error loading spaCy model: {e}. Linguistic features will be limited.")
            self.nlp = None
    
    def extract_enhanced_features(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive features from a text segment.
        
        Args:
            segment: Text segment with existing features
            
        Returns:
            Enhanced segment with additional features
        """
        text = segment.get("text", "")
        if isinstance(text, list):
            text = " ".join(text)
        
        if not text.strip():
            return segment
        
        # Get existing features
        existing_features = segment.get("features", {})
        
        # Extract new features
        enhanced_features = existing_features.copy()
        
        # Linguistic features
        enhanced_features.update(self._extract_linguistic_features(text))
        
        # Statistical features
        enhanced_features.update(self._extract_statistical_features(text))
        
        # Temporal features (enhance existing)
        enhanced_features.update(self._extract_temporal_features(text, existing_features))
        
        # Sustainability features
        enhanced_features.update(self._extract_sustainability_features(text))
        
        # Discourse features (enhance existing)
        enhanced_features.update(self._extract_discourse_features(text, existing_features))
        
        # Noun phrase features (enhance existing)
        enhanced_features.update(self._extract_noun_phrase_features(text, existing_features))
        
        # Update segment
        segment["features"] = enhanced_features
        
        return segment
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features using spaCy."""
        if not self.nlp:
            return {
                "pos_distribution": {},
                "pos_ratios": {},
                "named_entities": [],
                "entity_types": {},
                "dependency_counts": {},
                "avg_token_length": 0.0,
                "complex_sentences": 0
            }

        doc = self.nlp(text)
        
        # POS tag distribution
        pos_counts = Counter(token.pos_ for token in doc if not token.is_space)
        total_tokens = sum(pos_counts.values())
        
        pos_ratios = {}
        for pos in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]:
            pos_ratios[f"{pos.lower()}_ratio"] = pos_counts.get(pos, 0) / max(total_tokens, 1)
        
        # Named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entity_types = Counter(ent.label_ for ent in doc.ents)
        
        # Dependency features
        dep_counts = Counter(token.dep_ for token in doc)
        
        return {
            "pos_distribution": dict(pos_counts),
            "pos_ratios": pos_ratios,
            "named_entities": entities,
            "entity_types": dict(entity_types),
            "dependency_counts": dict(dep_counts),
            "avg_token_length": np.mean([len(token.text) for token in doc if not token.is_space]),
            "complex_sentences": len([sent for sent in doc.sents if len(sent) > 20])
        }
    
    def _extract_statistical_features(self, text: str) -> Dict[str, Any]:
        """Extract statistical text features."""
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Readability features
        avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Punctuation analysis
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        
        # Capitalization
        uppercase_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        capitalized_words = len(re.findall(r'\b[A-Z][a-z]+\b', text))
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "punctuation_ratio": punctuation_count / max(char_count, 1),
            "question_marks": question_marks,
            "exclamation_marks": exclamation_marks,
            "uppercase_words": uppercase_words,
            "capitalized_words": capitalized_words,
            "lexical_diversity": len(set(text.lower().split())) / max(word_count, 1)
        }
    
    def _extract_temporal_features(self, text: str, existing_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and enhance temporal features."""
        temporal_features = {
            "temporal_context": existing_features.get("temporal_context", "unknown"),
            "temporal_indicators": {
                "present_markers": [],
                "future_markers": [],
                "past_markers": []
            }
        }
        
        text_lower = text.lower()
        
        # Find temporal patterns
        for pattern in self.temporal_patterns["present"]:
            matches = re.findall(pattern, text_lower)
            temporal_features["temporal_indicators"]["present_markers"].extend(matches)
        
        for pattern in self.temporal_patterns["future"]:
            matches = re.findall(pattern, text_lower)
            temporal_features["temporal_indicators"]["future_markers"].extend(matches)
        
        # Past markers
        past_patterns = [
            r"\b(hier|autrefois|jadis|anciennement)\b",
            r"\b(l'année dernière|le mois dernier|la semaine dernière)\b",
            r"\b(dans le passé|auparavant|précédemment)\b"
        ]
        
        for pattern in past_patterns:
            matches = re.findall(pattern, text_lower)
            temporal_features["temporal_indicators"]["past_markers"].extend(matches)
        
        # Temporal confidence score
        present_count = len(temporal_features["temporal_indicators"]["present_markers"])
        future_count = len(temporal_features["temporal_indicators"]["future_markers"])
        past_count = len(temporal_features["temporal_indicators"]["past_markers"])
        
        total_temporal = present_count + future_count + past_count
        
        if total_temporal > 0:
            temporal_features["temporal_confidence"] = {
                "present": present_count / total_temporal,
                "future": future_count / total_temporal,
                "past": past_count / total_temporal
            }
        else:
            temporal_features["temporal_confidence"] = {
                "present": 0.0, "future": 0.0, "past": 0.0
            }
        
        return temporal_features
    
    def _extract_sustainability_features(self, text: str) -> Dict[str, Any]:
        """Extract sustainability-related features."""
        text_lower = text.lower()
        
        sustainability_scores = {}
        sustainability_terms = {}
        
        for category, terms in self.sustainability_indicators.items():
            found_terms = []
            score = 0
            
            for term in terms:
                count = text_lower.count(term)
                if count > 0:
                    found_terms.append({"term": term, "count": count})
                    score += count
            
            sustainability_scores[f"{category}_score"] = score
            sustainability_terms[f"{category}_terms"] = found_terms
        
        # Overall sustainability score
        total_score = sum(sustainability_scores.values())
        
        return {
            "sustainability_scores": sustainability_scores,
            "sustainability_terms": sustainability_terms,
            "total_sustainability_score": total_score,
            "sustainability_density": total_score / max(len(text.split()), 1)
        }
    
    def _extract_discourse_features(self, text: str, existing_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and enhance discourse features."""
        existing_markers = existing_features.get("discourse_markers", [])
        
        # Additional discourse patterns
        discourse_patterns = {
            "contrast": [r"\b(mais|cependant|néanmoins|toutefois|pourtant)\b"],
            "cause": [r"\b(parce que|car|puisque|étant donné)\b"],
            "consequence": [r"\b(donc|ainsi|par conséquent|c'est pourquoi)\b"],
            "addition": [r"\b(de plus|en outre|également|aussi)\b"],
            "temporal": [r"\b(d'abord|ensuite|puis|enfin|finalement)\b"]
        }
        
        discourse_analysis = {
            "discourse_markers": existing_markers,
            "discourse_types": {},
            "discourse_density": 0
        }
        
        text_lower = text.lower()
        total_markers = 0
        
        for marker_type, patterns in discourse_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            
            discourse_analysis["discourse_types"][marker_type] = count
            total_markers += count
        
        discourse_analysis["discourse_density"] = total_markers / max(len(text.split()), 1)
        
        return discourse_analysis
    
    def _extract_noun_phrase_features(self, text: str, existing_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and enhance noun phrase features."""
        existing_phrases = existing_features.get("noun_phrases", [])
        
        # Analyze existing noun phrases
        phrase_analysis = {
            "noun_phrases": existing_phrases,
            "phrase_count": len(existing_phrases),
            "unique_phrases": len(set(existing_phrases)),
            "avg_phrase_length": 0,
            "phrase_complexity": 0
        }
        
        if existing_phrases:
            phrase_lengths = [len(phrase.split()) for phrase in existing_phrases]
            phrase_analysis["avg_phrase_length"] = np.mean(phrase_lengths)
            phrase_analysis["phrase_complexity"] = len([p for p in existing_phrases if len(p.split()) > 2])
        
        # Extract additional noun phrases using spaCy (if available)
        spacy_noun_phrases = []
        if self.nlp:
            doc = self.nlp(text)
            spacy_noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        # Combine and deduplicate
        all_phrases = list(set(existing_phrases + spacy_noun_phrases))
        phrase_analysis["enhanced_noun_phrases"] = all_phrases
        phrase_analysis["total_phrase_count"] = len(all_phrases)
        
        return phrase_analysis
    
    def process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple segments to extract enhanced features.
        
        Args:
            segments: List of text segments
            
        Returns:
            Segments with enhanced features
        """
        if not segments:
            return []
        
        processed_segments = []
        
        for segment in segments:
            try:
                enhanced_segment = self.extract_enhanced_features(segment)
                processed_segments.append(enhanced_segment)
            except Exception as e:
                logger.error(f"Error processing segment {segment.get('segment_id', 'unknown')}: {e}")
                processed_segments.append(segment)  # Keep original if processing fails
        
        logger.info(f"Enhanced features for {len(processed_segments)} segments")
        return processed_segments
