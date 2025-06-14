from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Dict, Any, List

from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    transcript: str  # Raw transcript text
    preprocessed_data: Dict[str, Any]  # Data from preprocessing pipeline
    segments: Annotated[list, operator.add]  # Extracted segments
    analysis_results: Annotated[list, operator.add]  # Analysis results for each segment
    final_results: list  # Final formatted results
    max_segments: int  # Maximum number of segments to process


class SegmentationState(TypedDict):
    segments: list  # List of segment dictionaries with metadata


class AnalysisResult(TypedDict):
    segment_id: str
    concepts_2nd_ordre: str
    items_1er_ordre_reformule: str
    items_1er_ordre_origine: str
    details: str
    synthese: str
    periode: str
    theme: str
    code_spe: str
    imaginaire: str
