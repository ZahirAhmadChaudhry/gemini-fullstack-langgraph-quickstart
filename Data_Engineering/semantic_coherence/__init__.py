"""
Semantic Coherence Package

This package contains modules for analyzing and ensuring semantic coherence
in French text segments.
"""

from .discourse_analyzer import DiscourseMarker, DiscourseAnalyzer
from .thematic_tracker import ThematicTracker
from .segment_validator import SegmentValidator
from .coherence_measurer import SemanticCoherenceMeasurer

__all__ = [
    'DiscourseMarker',
    'DiscourseAnalyzer',
    'ThematicTracker',
    'SegmentValidator',
    'SemanticCoherenceMeasurer'
]