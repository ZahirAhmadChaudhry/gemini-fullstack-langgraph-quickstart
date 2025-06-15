"""Module for measuring and ensuring semantic coherence in text segments."""

# This file now serves as a bridge to maintain backward compatibility
# All actual functionality has been moved to the semantic_coherence package

from semantic_coherence.discourse_analyzer import DiscourseMarker, DiscourseAnalyzer
from semantic_coherence.thematic_tracker import ThematicTracker
from semantic_coherence.segment_validator import SegmentValidator
from semantic_coherence.coherence_measurer import SemanticCoherenceMeasurer

__all__ = [
    'DiscourseMarker', 
    'DiscourseAnalyzer',
    'ThematicTracker',
    'SegmentValidator',
    'SemanticCoherenceMeasurer'
]