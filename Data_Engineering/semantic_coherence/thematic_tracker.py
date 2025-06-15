"""Module for tracking thematic continuity in text segments."""

from typing import List, Dict, Tuple, Set
import spacy
from collections import defaultdict
import logging
from .discourse_analyzer import DiscourseAnalyzer

logger = logging.getLogger(__name__)

class ThematicTracker:
    """Tracks thematic continuity in text segments."""
    
    def __init__(self, nlp_model):
        """Initialize the thematic tracker with a given spaCy model."""
        self.nlp = nlp_model
    
    def measure_thematic_similarity(self, segment1: List[str], segment2: List[str]) -> float:
        """Measure thematic similarity between two segments."""
        # Convert segments to single strings
        text1 = " ".join(segment1)
        text2 = " ".join(segment2)
        
        # Process with spaCy
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        # Calculate similarity using spaCy's vector similarity
        vector_similarity = doc1.similarity(doc2)
        
        # Calculate lexical overlap
        overlap_similarity = self._calculate_lexical_overlap(doc1, doc2)
        
        # Track named entities and shared references
        entity_similarity = self._calculate_entity_similarity(doc1, doc2)
        
        # Combine the different similarity measures
        # Give higher weight to entity similarity as research indicates NEs are key anchors
        combined_similarity = (
            vector_similarity * 0.4 + 
            overlap_similarity * 0.3 + 
            entity_similarity * 0.3
        )
        
        return combined_similarity
    
    def _calculate_lexical_overlap(self, doc1, doc2) -> float:
        """Calculate lexical overlap between two documents."""
        # Extract content words (nouns, verbs, adjectives)
        content_words1 = set(
            token.lemma_.lower() for token in doc1 
            if token.pos_ in ('NOUN', 'VERB', 'ADJ') 
            and not token.is_stop and len(token.lemma_) > 1
        )
        
        content_words2 = set(
            token.lemma_.lower() for token in doc2 
            if token.pos_ in ('NOUN', 'VERB', 'ADJ') 
            and not token.is_stop and len(token.lemma_) > 1
        )
        
        # Calculate Jaccard similarity
        if not content_words1 and not content_words2:
            return 0.5  # Neutral score
        
        intersection = content_words1.intersection(content_words2)
        union = content_words1.union(content_words2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
    
    def _calculate_entity_similarity(self, doc1, doc2) -> float:
        """Calculate similarity based on shared named entities."""
        # Extract named entities
        entities1 = set(ent.text.lower() for ent in doc1.ents)
        entities2 = set(ent.text.lower() for ent in doc2.ents)
        
        # Add potential coreferent mentions (definite noun phrases)
        for np in doc1.noun_chunks:
            if np[0].pos_ == 'DET' and np[0].text.lower() in ('le', 'la', 'les', 'l\''):
                entities1.add(np.text.lower())
                
        for np in doc2.noun_chunks:
            if np[0].pos_ == 'DET' and np[0].text.lower() in ('le', 'la', 'les', 'l\''):
                entities2.add(np.text.lower())
        
        # Calculate similarity based on shared entities
        if not entities1 and not entities2:
            return 0.5  # Neutral score if no entities
            
        if not entities1 or not entities2:
            return 0.25  # Low score if entities in only one segment
        
        intersection = entities1.intersection(entities2)
        union = entities1.union(entities2)
        
        return len(intersection) / len(union)
    
    def extract_key_terms(self, segment: List[str]) -> List[Tuple[str, float]]:
        """Extract key terms and their importance from a segment."""
        text = " ".join(segment)
        doc = self.nlp(text)
        
        # Count term frequencies with weighting based on POS
        term_freq = defaultdict(float)
        for token in doc:
            # Consider nouns, verbs, and adjectives with different weights
            if token.pos_ == 'NOUN' and not token.is_stop:
                term_freq[token.lemma_] += 1.2  # Higher weight for nouns
            elif token.pos_ == 'PROPN':  # Proper nouns (names)
                term_freq[token.lemma_] += 1.5  # Highest weight for proper nouns
            elif token.pos_ == 'VERB' and not token.is_stop:
                term_freq[token.lemma_] += 1.0
            elif token.pos_ == 'ADJ' and not token.is_stop:
                term_freq[token.lemma_] += 0.8
        
        # Add named entities with high weights
        for ent in doc.ents:
            term_freq[ent.text] += 2.0  # Named entities are strong thematic indicators
        
        # Normalize frequencies
        max_freq = max(term_freq.values()) if term_freq else 1.0
        return [(term, freq/max_freq) for term, freq in term_freq.items()]
    
    def detect_topic_shift(self, prev_segment: List[str], current_segment: List[str]) -> float:
        """
        Detect if there's a topic shift between segments.
        Returns a score between 0.0 (no shift) and 1.0 (complete shift).
        """
        # Check similarity between segments
        similarity = self.measure_thematic_similarity(prev_segment, current_segment)
        
        # Transform to shift score (invert similarity)
        shift_score = 1.0 - similarity
        
        # Check for explicit topic-shifting discourse markers
        current_text = " ".join(current_segment)
        discourse_analyzer = DiscourseAnalyzer()
        markers = discourse_analyzer.find_discourse_markers(current_text)
        
        # If topic shift markers are present, increase the shift score
        for marker in markers:
            if marker.type == "topic_shift" and marker.position == "start":
                shift_score = min(1.0, shift_score + 0.3)  # Boost the shift score
                break
        
        # Check if there are new named entities introduced
        prev_text = " ".join(prev_segment)
        prev_doc = self.nlp(prev_text)
        current_doc = self.nlp(current_text)
        
        prev_entities = set(ent.text.lower() for ent in prev_doc.ents)
        current_entities = set(ent.text.lower() for ent in current_doc.ents)
        
        # Calculate percentage of new entities
        if current_entities:
            new_entities = current_entities - prev_entities
            new_entity_ratio = len(new_entities) / len(current_entities)
            
            # If many new entities are introduced, boost the shift score
            if new_entity_ratio > 0.5:
                shift_score = min(1.0, shift_score + 0.2)
        
        return shift_score
    
    def create_lexical_chains(self, sentences: List[str]) -> Dict[str, List[int]]:
        """
        Create lexical chains across sentences to track thematic continuity.
        Returns a dictionary mapping key terms to the indices of sentences they appear in.
        """
        chains = defaultdict(list)
        
        for i, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            
            # Track content words and named entities
            for token in doc:
                if token.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ') and not token.is_stop:
                    chains[token.lemma_.lower()].append(i)
            
            # Add named entities (potentially multi-word)
            for ent in doc.ents:
                chains[ent.text.lower()].append(i)
        
        # Filter out chains that only appear in one sentence
        return {term: mentions for term, mentions in chains.items() if len(mentions) > 1}