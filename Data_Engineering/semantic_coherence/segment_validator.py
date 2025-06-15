"""Module for validating and scoring text segments for coherence."""

from typing import List, Dict, Any
import logging
from .discourse_analyzer import DiscourseAnalyzer, DiscourseMarker
from .thematic_tracker import ThematicTracker

logger = logging.getLogger(__name__)

class SegmentValidator:
    """Validates and scores text segments for coherence."""
    
    def __init__(self, nlp_model=None):
        """Initialize the segment validator."""
        self.min_lines = 2
        self.max_lines = 10
        self.discourse_analyzer = DiscourseAnalyzer()
        self.thematic_tracker = ThematicTracker(nlp_model)
    
    def validate_segment(self, segment: List[str]) -> Dict[str, Any]:
        """Validate a segment and return metrics."""
        # Check size constraints
        size_score = self._check_size(segment)
        
        # Check discourse markers
        joined_text = " ".join(segment)
        discourse_markers = self.discourse_analyzer.find_discourse_markers(joined_text)
        marker_score = self._evaluate_markers(discourse_markers)
        
        # Check thematic coherence
        thematic_score = self._measure_thematic_coherence(segment)
        
        # Calculate overall score with weighted components
        # Based on research, discourse markers and thematic coherence are more important
        total_score = (
            size_score * 0.2 +  # Size constraints (20%)
            marker_score * 0.4 +  # Discourse markers (40%)
            thematic_score * 0.4  # Thematic coherence (40%)
        )
        
        # Create a detailed analysis for debugging
        marker_types = set(m.type for m in discourse_markers)
        has_priority_marker = any(m.type in ["sequential", "conclusive", "topic_shift"] for m in discourse_markers)
        
        return {
            "size_score": size_score,
            "marker_score": marker_score,
            "thematic_score": thematic_score,
            "total_score": total_score,
            "has_priority_marker": has_priority_marker,
            "marker_types": list(marker_types),
            "discourse_markers": [
                {
                    "text": m.text,
                    "type": m.type,
                    "strength": m.strength,
                    "position": m.position
                }
                for m in discourse_markers
            ]
        }
    
    def _check_size(self, segment: List[str]) -> float:
        """Check if segment size is within bounds."""
        size = len(segment)
        if size < self.min_lines:
            return 0.0  # Too small
        elif size > self.max_lines:
            return 0.0  # Too large
        else:
            # Graduated scoring based on optimal size
            # Research shows medium-sized segments (4-6 lines) tend to maintain optimal coherence
            if size <= 3:  # Small but valid
                return 0.7
            elif 4 <= size <= 6:  # Optimal range
                return 1.0
            else:  # Large but valid (7-10)
                # Gradually decrease score as size increases
                return 0.9 - ((size - 6) * 0.1)
    
    def _evaluate_markers(self, markers: List[DiscourseMarker]) -> float:
        """Evaluate the quality of discourse markers in the segment."""
        if not markers:
            return 0.5  # Neutral score for no markers
        
        # Calculate weighted score based on marker strength and position
        total_weight = 0.0
        score = 0.0
        
        # First pass: identify highest priority markers
        # Research shows specific types are more important for segmentation
        priority_markers = [m for m in markers if m.type in ["sequential", "conclusive", "topic_shift"]]
        
        if priority_markers:
            # If we have priority markers, focus scoring on them
            for marker in priority_markers:
                # Apply position multipliers
                position_multiplier = 1.0
                if marker.position == "start" and marker.type in ["sequential", "topic_shift"]:
                    position_multiplier = 1.5  # Boost importance of starting markers
                elif marker.position == "end" and marker.type == "conclusive":
                    position_multiplier = 1.4  # Boost importance of ending markers
                
                weight = marker.strength * position_multiplier
                score += weight * marker.strength
                total_weight += weight
            
            # Return score based on priority markers
            return min(1.0, score / total_weight if total_weight > 0 else 0.5)
        else:
            # Fall back to general marker evaluation
            for marker in markers:
                weight = marker.strength
                if marker.position == "start":
                    weight *= 1.3
                elif marker.position == "end":
                    weight *= 1.2
                
                score += weight * marker.strength
                total_weight += weight
            
            return score / total_weight if total_weight > 0 else 0.5
    
    def _measure_thematic_coherence(self, segment: List[str]) -> float:
        """Measure thematic coherence within a segment."""
        # Extract key terms
        key_terms = self.thematic_tracker.extract_key_terms(segment)
        
        if not key_terms:
            return 0.5  # Neutral score if no key terms found
        
        # Calculate coherence based on term distribution
        term_weights = [weight for _, weight in key_terms]
        term_based_coherence = sum(term_weights) / len(term_weights)
        
        # If segment has multiple sentences, check inter-sentence similarity
        if len(segment) > 1:
            # Create lexical chains to track thematic continuity
            chains = self.thematic_tracker.create_lexical_chains(segment)
            
            # Measure chain density and coverage
            chain_coherence = 0.0
            if chains:
                # Calculate how many sentences each chain covers, on average
                chain_coverage = [len(sentences) / len(segment) for sentences in chains.values()]
                average_coverage = sum(chain_coverage) / len(chain_coverage) if chain_coverage else 0.0
                
                # Higher coverage = higher coherence
                chain_coherence = average_coverage
                
                # Compare adjacent sentences
                similarities = []
                for i in range(len(segment) - 1):
                    sim = self.thematic_tracker.measure_thematic_similarity(
                        [segment[i]], [segment[i + 1]]
                    )
                    similarities.append(sim)
                
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                
                # Combine term-based, chain-based, and similarity-based scores
                # Weighted based on research showing importance of chain continuity
                coherence = (
                    term_based_coherence * 0.3 + 
                    chain_coherence * 0.4 + 
                    avg_similarity * 0.3
                )
                return coherence
        
        # For single-sentence segments, just use term-based coherence
        return term_based_coherence