"""Module for analyzing discourse structure and markers in text."""

from typing import List, Tuple, Dict
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DiscourseMarker:
    """Represents a discourse marker with its type and strength."""
    text: str
    type: str  # 'sequential', 'conclusive', etc.
    strength: float  # 0.0 to 1.0
    position: str  # 'start', 'middle', 'end'

class DiscourseAnalyzer:
    """Analyzes discourse structure and markers in text."""
    
    def __init__(self):
        """Initialize the discourse analyzer with marker categories."""
        # Structural organizers (high importance for segmentation)
        self.sequential_markers = {
            "premièrement": 1.0, "d'abord": 0.9, "en premier lieu": 0.9,
            "deuxièmement": 1.0, "ensuite": 0.8, "puis": 0.7,
            "enfin": 0.9, "en dernier lieu": 0.9, "troisièmement": 1.0,
            "pour commencer": 0.9, "dans un premier temps": 0.9,
            "dans un second temps": 0.9, "finalement": 0.85
        }
        
        self.conclusive_markers = {
            "en conclusion": 1.0, "pour conclure": 1.0,
            "finalement": 0.8, "en somme": 0.9,
            "donc": 0.7, "ainsi": 0.7, "au final": 0.85,
            "en résumé": 0.9, "pour résumer": 0.9,
            "bref": 0.8, "en définitive": 0.9
        }
        
        # Topic shifters (high importance)
        self.topic_shift_markers = {
            "à propos": 0.9, "au fait": 0.9, 
            "d'ailleurs": 0.85, "par ailleurs": 0.85,
            "en ce qui concerne": 0.8, "quant à": 0.8,
            "pour ce qui est de": 0.8, "concernant": 0.75,
            "à ce sujet": 0.75, "sur ce point": 0.75
        }
        
        # Connectors (moderate importance)
        self.contrastive_markers = {
            "cependant": 0.8, "mais": 0.6, "toutefois": 0.8,
            "par contre": 0.8, "en revanche": 0.8,
            "néanmoins": 0.8, "pourtant": 0.75,
            "malgré cela": 0.7, "malgré tout": 0.7,
            "en dépit de": 0.7
        }
        
        self.causal_markers = {
            "parce que": 0.65, "car": 0.65,
            "à cause de": 0.7, "en raison de": 0.7,
            "puisque": 0.6, "grâce à": 0.6,
            "c'est pourquoi": 0.75, "par conséquent": 0.75,
            "en conséquence": 0.75, "de ce fait": 0.7
        }
        
        self.additive_markers = {
            "de plus": 0.7, "en outre": 0.7, "également": 0.6,
            "par ailleurs": 0.7, "aussi": 0.5,
            "de surcroît": 0.65, "en plus": 0.6,
            "non seulement": 0.65, "de même": 0.6
        }
        
        # Reformulators (lower importance for boundaries)
        self.reformulation_markers = {
            "c'est-à-dire": 0.5, "autrement dit": 0.5,
            "en d'autres termes": 0.5, "je veux dire": 0.45,
            "à savoir": 0.5, "en fait": 0.4,
            "cela signifie": 0.45, "pour être précis": 0.5
        }
        
        # Conversational/phatic markers (lowest importance for boundaries)
        self.conversational_markers = {
            "voilà": 0.4, "bon": 0.3, "ben": 0.3,
            "hein": 0.2, "tu vois": 0.25, "tu sais": 0.25,
            "euh": 0.1, "quoi": 0.3, "bah": 0.3,
            "eh bien": 0.4, "écoute": 0.45, "vous savez": 0.3
        }
        
        # Common marker collocations that typically occur within segments
        self.internal_combinations = [
            ("bon", "ben"), ("et", "puis"), ("et", "bon"),
            ("alors", "bon"), ("enfin", "bref"), ("mais", "aussi"),
            ("en", "fait"), ("c'est", "pourquoi")
        ]
        
        # Combinations that likely indicate segment boundaries
        self.boundary_combinations = [
            ("mais", "cependant"), ("premièrement", "ensuite"),
            ("d'abord", "enfin"), ("ensuite", "en conclusion"),
            ("alors", "donc"), ("non seulement", "mais aussi")
        ]
        
        # Compile regex patterns for marker detection
        self.marker_patterns = {}
        for marker_type, markers in [
            ("sequential", self.sequential_markers),
            ("conclusive", self.conclusive_markers),
            ("topic_shift", self.topic_shift_markers),
            ("contrastive", self.contrastive_markers),
            ("causal", self.causal_markers),
            ("additive", self.additive_markers),
            ("reformulation", self.reformulation_markers),
            ("conversational", self.conversational_markers)
        ]:
            for marker in markers:
                pattern = re.compile(rf'\b{re.escape(marker)}\b', re.IGNORECASE)
                self.marker_patterns[marker] = (pattern, marker_type)
    
    def find_discourse_markers(self, text: str) -> List[DiscourseMarker]:
        """Find all discourse markers in the text with their types and positions."""
        markers = []
        
        # Check for markers
        for marker, (pattern, marker_type) in self.marker_patterns.items():
            matches = pattern.finditer(text.lower())
            for match in matches:
                # Determine position more precisely
                start_pos = match.start()
                text_before = text[:start_pos].strip()
                
                # More sophisticated position detection
                if not text_before or text_before.endswith(('.', '!', '?', ':', '\n')):
                    position = "start"  # Marker at start of a sentence or after punctuation
                elif start_pos >= len(text) - len(marker) - 20:
                    position = "end"  # Marker near end
                else:
                    position = "middle"
                
                # Get strength based on marker type
                if marker_type == "sequential":
                    strength = self.sequential_markers.get(marker, 0.5)
                elif marker_type == "conclusive":
                    strength = self.conclusive_markers.get(marker, 0.5)
                elif marker_type == "topic_shift":
                    strength = self.topic_shift_markers.get(marker, 0.5)
                elif marker_type == "contrastive":
                    strength = self.contrastive_markers.get(marker, 0.5)
                elif marker_type == "causal":
                    strength = self.causal_markers.get(marker, 0.5)
                elif marker_type == "additive":
                    strength = self.additive_markers.get(marker, 0.5)
                elif marker_type == "reformulation":
                    strength = self.reformulation_markers.get(marker, 0.5)
                else:  # conversational
                    strength = self.conversational_markers.get(marker, 0.5)
                
                # Position-based weighting
                if position == "start" and marker_type in ["sequential", "conclusive", "topic_shift"]:
                    strength *= 1.5  # Significantly boost structural markers at start
                elif position == "end" and marker_type == "conclusive":
                    strength *= 1.3  # Boost conclusive markers at end
                elif position == "middle" and marker_type in ["reformulation", "conversational"]:
                    strength *= 0.7  # Reduce weight of certain markers in middle position
                
                markers.append(DiscourseMarker(
                    text=marker,
                    type=marker_type,
                    strength=strength,
                    position=position
                ))
        
        # Check for marker combinations
        self._process_marker_combinations(text, markers)
        
        return markers
    
    def _process_marker_combinations(self, text: str, markers: List[DiscourseMarker]):
        """Process marker combinations to adjust strengths."""
        text_lower = text.lower()
        
        # Check for internal combinations (reduce boundary strength)
        for m1, m2 in self.internal_combinations:
            if re.search(rf'\b{re.escape(m1)}\b.*\b{re.escape(m2)}\b', text_lower, re.DOTALL):
                # Reduce strength of these markers when they appear together
                for marker in markers:
                    if marker.text.lower() == m1 or marker.text.lower() == m2:
                        marker.strength *= 0.7
        
        # Check for boundary combinations (increase boundary strength)
        for m1, m2 in self.boundary_combinations:
            if re.search(rf'\b{re.escape(m1)}\b.*\b{re.escape(m2)}\b', text_lower, re.DOTALL):
                # Increase strength of these markers when they appear together
                for marker in markers:
                    if marker.text.lower() == m1 or marker.text.lower() == m2:
                        marker.strength *= 1.3
    
    def identify_marker_type(self, text: str) -> Tuple[bool, str]:
        """
        Identify the type of discourse marker in a segment.
        Returns (has_marker, marker_type).
        """
        text_lower = text.lower()
        
        # First check for sequential and conclusive markers (highest priority)
        for marker in self.sequential_markers:
            if re.search(rf'\b{re.escape(marker)}\b', text_lower):
                return True, "sequential"
        
        for marker in self.conclusive_markers:
            if re.search(rf'\b{re.escape(marker)}\b', text_lower):
                return True, "conclusive"
        
        # Check for topic shift markers
        for marker in self.topic_shift_markers:
            if re.search(rf'\b{re.escape(marker)}\b', text_lower):
                return True, "topic_shift"
                
        # Check for contrastive and causal markers
        for marker in self.contrastive_markers:
            if re.search(rf'\b{re.escape(marker)}\b', text_lower):
                return True, "contrastive"
        
        for marker in self.causal_markers:
            if re.search(rf'\b{re.escape(marker)}\b', text_lower):
                return True, "causal"
                
        # Check for additive and reformulation markers
        for marker in self.additive_markers:
            if re.search(rf'\b{re.escape(marker)}\b', text_lower):
                return True, "additive"
        
        for marker in self.reformulation_markers:
            if re.search(rf'\b{re.escape(marker)}\b', text_lower):
                return True, "reformulation"
        
        # Finally check for conversational markers
        for marker in self.conversational_markers:
            if re.search(rf'\b{re.escape(marker)}\b', text_lower):
                return True, "conversational"
        
        return False, ""