#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML-Ready Data Formatter for French Transcripts

This module prepares processed transcript segments for machine learning
by formatting them in a standardized JSON structure.
"""

import json
import uuid
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class MlReadyFormatter:
    """
    Enhanced ML-Ready Data Formatter for French Transcripts

    Creates standardized JSON output optimized for ML pipelines to generate
    data equivalent to the human-annotated reference format (data.json).

    Target output format:
    - "Concepts de 2nd ordre": High-level conceptual categories
    - "Items de 1er ordre reformulé": Refined first-order concepts
    - "Items de 1er ordre (intitulé d'origine)": Original first-order labels
    - "Détails": Raw transcript segments
    - "Période": Temporal context (2050.0)
    - "Thème": Thematic categories (Performance, Légitimité)
    - "Code spé": Specialized tension codes
    """

    def __init__(self, output_dir: str = "ml_ready_data"):
        """
        Initialize the enhanced formatter.

        Args:
            output_dir: Directory to save formatted output files
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize thematic classification patterns based on data.json analysis
        self._init_thematic_patterns()
        self._init_tension_patterns()
        self._init_temporal_patterns()

    def _init_thematic_patterns(self):
        """Initialize thematic classification patterns based on data.json analysis."""
        self.performance_indicators = [
            "performance", "efficacité", "croissance", "résultats", "indicateurs",
            "mesure", "évaluation", "productivité", "optimisation", "bénéfices",
            "rentabilité", "compétitivité", "innovation", "développement"
        ]

        self.legitimacy_indicators = [
            "légitimité", "éthique", "responsabilité", "durabilité", "social",
            "environnemental", "équitable", "justice", "transparence", "confiance",
            "acceptabilité", "crédibilité", "authenticité", "intégrité"
        ]

    def _init_tension_patterns(self):
        """Initialize tension detection patterns based on data.json opposing concepts."""
        self.tension_patterns = {
            "accumulation_partage": {
                "accumulation": ["accumulation", "accumul", "propriété", "capital", "richesse", "profit"],
                "partage": ["partage", "partag", "redistribution", "commun", "collectif", "mutualisation"]
            },
            "croissance_decroissance": {
                "croissance": ["croissance", "expansion", "développement", "augmentation"],
                "décroissance": ["décroissance", "limitation", "réduction", "stabilisation"]
            },
            "individuel_collectif": {
                "individuel": ["individuel", "personnel", "privé", "liberté"],
                "collectif": ["collectif", "commun", "société", "ensemble"]
            },
            "court_terme_long_terme": {
                "court_terme": ["immédiat", "court terme", "urgent", "rapide"],
                "long_terme": ["long terme", "durable", "pérenne", "futur"]
            },
            "local_global": {
                "local": ["local", "proximité", "territoire", "régional"],
                "global": ["global", "mondial", "international", "planétaire"]
            }
        }

    def _init_temporal_patterns(self):
        """Initialize enhanced temporal detection patterns."""
        self.temporal_indicators = {
            "2023": ["2023", "aujourd'hui", "actuellement", "maintenant", "présent"],
            "2050": ["2050", "futur", "demain", "avenir", "horizon", "perspective"]
        }

    def format_segments(self,
                       segments: List[Dict[str, Any]],
                       source_file: str,
                       nlp_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format segments into enhanced ML-ready structure optimized for generating
        data equivalent to the human-annotated reference format.

        Args:
            segments: List of segment dictionaries from preprocessor
            source_file: Source file name
            nlp_results: Optional NLP results with tokens, lemmas, etc.

        Returns:
            Dictionary with enhanced ML-ready structured data for target format generation
        """
        formatted_data = {
            "source_file": source_file,
            "processed_timestamp": datetime.now().isoformat(),
            "ml_target_format": "data_json_compatible",
            "segments": []
        }
        
        for i, segment in enumerate(segments):
            # Extract segment text
            if "text" in segment and isinstance(segment["text"], list):
                text = " ".join(segment["text"])
            elif "segment_text" in segment:
                text = segment["segment_text"]
            else:
                text = ""

            # Extract or generate segment ID
            segment_id = segment.get("id", f"{os.path.splitext(source_file)[0]}_seg_{i+1:03d}")

            # Extract position information
            position = {
                "start": segment.get("start_sentence_index", i),
                "end": segment.get("end_sentence_index", i + len(text.split()) // 10)  # Rough estimate if not provided
            }

            # Count sentences and words
            sentence_count = len(segment.get("text", [])) if isinstance(segment.get("text", []), list) else 1
            word_count = len(text.split())

            # Enhanced feature extraction for ML pipeline compatibility
            features = {
                # Core linguistic features
                "temporal_context": self._get_enhanced_temporal_context(segment, text),
                "discourse_markers": self._extract_discourse_markers(segment),
                "sentence_count": sentence_count,
                "word_count": word_count,

                # Enhanced thematic classification features
                "thematic_indicators": self._extract_thematic_indicators(text),
                "tension_patterns": self._detect_tension_patterns(text),
                "conceptual_markers": self._extract_conceptual_markers(text),

                # ML-ready metadata for target format generation
                "ml_features": {
                    "performance_score": self._calculate_performance_score(text),
                    "legitimacy_score": self._calculate_legitimacy_score(text),
                    "temporal_period": self._determine_temporal_period(segment, text),
                    "tension_indicators": self._extract_tension_indicators(text),
                    "conceptual_complexity": self._assess_conceptual_complexity(text)
                }
            }

            # Extract noun phrases if available
            if nlp_results and "noun_phrases" in nlp_results:
                features["noun_phrases"] = self._find_noun_phrases_in_segment(nlp_results["noun_phrases"], text)

            # Create the enhanced formatted segment
            formatted_segment = {
                "id": segment_id,
                "text": text,
                "features": features,
                "metadata": {
                    "source": source_file,
                    "segment_lines": sentence_count,
                    "position": position,
                    "ml_readiness_score": self._calculate_ml_readiness_score(features),
                    "target_format_compatibility": True
                }
            }

            formatted_data["segments"].append(formatted_segment)
        
        return formatted_data

    def _get_enhanced_temporal_context(self, segment: Dict[str, Any], text: str) -> str:
        """Enhanced temporal context detection for ML pipeline compatibility."""
        # Check segment-level temporal markers first
        if segment.get("present_context") or segment.get("temporal_markers", {}).get("2023_reference"):
            return "2023"
        elif segment.get("future_context") or segment.get("temporal_markers", {}).get("2050_reference"):
            return "2050"

        # Check text content for temporal indicators
        text_lower = text.lower()
        for period, indicators in self.temporal_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return period

        return "unknown"

    def _extract_thematic_indicators(self, text: str) -> Dict[str, float]:
        """Extract thematic classification indicators for ML processing."""
        text_lower = text.lower()

        performance_count = sum(1 for indicator in self.performance_indicators
                              if indicator in text_lower)
        legitimacy_count = sum(1 for indicator in self.legitimacy_indicators
                             if indicator in text_lower)

        total_words = len(text.split())

        return {
            "performance_density": performance_count / max(1, total_words / 100),
            "legitimacy_density": legitimacy_count / max(1, total_words / 100),
            "performance_indicators": performance_count,
            "legitimacy_indicators": legitimacy_count
        }

    def _detect_tension_patterns(self, text: str) -> Dict[str, Any]:
        """Detect opposing concept patterns for tension analysis."""
        text_lower = text.lower()
        detected_tensions = {}

        for tension_name, tension_dict in self.tension_patterns.items():
            side_a_count = sum(1 for term in tension_dict[list(tension_dict.keys())[0]]
                             if term in text_lower)
            side_b_count = sum(1 for term in tension_dict[list(tension_dict.keys())[1]]
                             if term in text_lower)

            if side_a_count > 0 or side_b_count > 0:
                detected_tensions[tension_name] = {
                    "side_a": side_a_count,
                    "side_b": side_b_count,
                    "tension_strength": abs(side_a_count - side_b_count),
                    "total_indicators": side_a_count + side_b_count
                }

        return detected_tensions

    def _extract_conceptual_markers(self, text: str) -> List[str]:
        """Extract conceptual markers that indicate high-level themes."""
        conceptual_markers = []
        text_lower = text.lower()

        # Economic model indicators
        economic_terms = ["économique", "économie", "modèle", "système", "capitalisme", "marché"]
        if any(term in text_lower for term in economic_terms):
            conceptual_markers.append("MODELES_SOCIO_ECONOMIQUES")

        # Organizational indicators
        org_terms = ["organisation", "entreprise", "structure", "management", "gouvernance"]
        if any(term in text_lower for term in org_terms):
            conceptual_markers.append("MODELES_ORGANISATIONNELS")

        # Environmental indicators
        env_terms = ["environnement", "climat", "écologie", "durable", "vert"]
        if any(term in text_lower for term in env_terms):
            conceptual_markers.append("MODELES_ENVIRONNEMENTAUX")

        return conceptual_markers

    def _get_temporal_context(self, segment: Dict[str, Any]) -> str:
        """Extract temporal context from segment."""
        if segment.get("present_context") or segment.get("temporal_markers", {}).get("2023_reference"):
            return "2023"
        elif segment.get("future_context") or segment.get("temporal_markers", {}).get("2050_reference"):
            return "2050"
        else:
            return "unknown"
    
    def _calculate_performance_score(self, text: str) -> float:
        """Calculate performance theme relevance score."""
        text_lower = text.lower()
        score = sum(1 for indicator in self.performance_indicators if indicator in text_lower)
        return min(1.0, score / 5.0)  # Normalize to 0-1 scale

    def _calculate_legitimacy_score(self, text: str) -> float:
        """Calculate legitimacy theme relevance score."""
        text_lower = text.lower()
        score = sum(1 for indicator in self.legitimacy_indicators if indicator in text_lower)
        return min(1.0, score / 5.0)  # Normalize to 0-1 scale

    def _determine_temporal_period(self, segment: Dict[str, Any], text: str) -> float:
        """Determine temporal period as numeric value for ML processing."""
        temporal_context = self._get_enhanced_temporal_context(segment, text)
        if temporal_context == "2023":
            return 2023.0
        elif temporal_context == "2050":
            return 2050.0
        else:
            return 2035.0  # Default middle value for unknown

    def _extract_tension_indicators(self, text: str) -> List[str]:
        """Extract specific tension indicators for ML classification."""
        tensions = self._detect_tension_patterns(text)
        return [tension_name for tension_name, data in tensions.items()
                if data["total_indicators"] > 0]

    def _assess_conceptual_complexity(self, text: str) -> float:
        """Assess conceptual complexity for ML feature engineering."""
        word_count = len(text.split())
        unique_concepts = len(self._extract_conceptual_markers(text))
        tension_count = len(self._extract_tension_indicators(text))

        # Simple complexity score based on multiple factors
        complexity = (unique_concepts * 0.4 + tension_count * 0.6) / max(1, word_count / 100)
        return min(1.0, complexity)

    def _calculate_ml_readiness_score(self, features: Dict[str, Any]) -> float:
        """Calculate overall ML readiness score for the segment."""
        scores = []

        # Check feature completeness
        if features.get("thematic_indicators"):
            scores.append(0.3)
        if features.get("tension_patterns"):
            scores.append(0.3)
        if features.get("conceptual_markers"):
            scores.append(0.2)
        if features.get("ml_features"):
            scores.append(0.2)

        return sum(scores)

    def _extract_discourse_markers(self, segment: Dict[str, Any]) -> List[str]:
        """Extract discourse markers from segment."""
        markers = []

        # Try different locations where markers might be stored
        if "discourse_marker_info" in segment and segment["discourse_marker_info"]:
            if isinstance(segment["discourse_marker_info"], list):
                markers.extend(segment["discourse_marker_info"])
            elif isinstance(segment["discourse_marker_info"], dict):
                for _, marker_list in segment["discourse_marker_info"].items():
                    if isinstance(marker_list, list):
                        markers.extend(marker_list)
                    else:
                        markers.append(str(marker_list))
            else:
                markers.append(str(segment["discourse_marker_info"]))

        # Check has_discourse_marker and discourse_marker_type
        if segment.get("has_discourse_marker") and segment.get("discourse_marker_type"):
            markers.append(segment["discourse_marker_type"])

        return list(set(markers))  # Remove duplicates
    
    def _find_noun_phrases_in_segment(self, all_phrases: List[str], segment_text: str) -> List[str]:
        """Find noun phrases that appear in the segment text."""
        return [phrase for phrase in all_phrases if phrase.lower() in segment_text.lower()]
    
    def save_to_file(self, formatted_data: Dict[str, Any], output_filename: Optional[str] = None) -> str:
        """
        Save formatted data to JSON file.
        
        Args:
            formatted_data: ML-ready data structure
            output_filename: Optional output filename, defaults to source file name with _ml_ready suffix
            
        Returns:
            Path to the saved file
        """
        if not output_filename:
            source_file = formatted_data["source_file"]
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            output_filename = f"{base_name}_ml_ready.json"
        
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved ML-ready data to {output_path}")
        return str(output_path)
