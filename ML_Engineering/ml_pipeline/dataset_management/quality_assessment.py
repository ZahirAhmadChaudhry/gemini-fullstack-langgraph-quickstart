"""
Quality Assessment module for evaluating dataset quality.

This module provides tools for assessing data quality, detecting outliers,
and generating data profiling reports.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re

# Configure logging
logger = logging.getLogger(__name__)

class QualityAssessment:
    """
    Dataset quality assessment and profiling system.
    """
    
    def __init__(self):
        """Initialize the quality assessment system."""
        pass
    
    def assess_segment_quality(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the quality of text segments.
        
        Args:
            segments: List of text segments
            
        Returns:
            Quality assessment report
        """
        if not segments:
            return {"error": "No segments provided"}
        
        logger.info(f"Assessing quality of {len(segments)} segments")
        
        quality_report = {
            "total_segments": len(segments),
            "completeness": self._assess_completeness(segments),
            "consistency": self._assess_consistency(segments),
            "validity": self._assess_validity(segments),
            "uniqueness": self._assess_uniqueness(segments),
            "outliers": self._detect_outliers(segments),
            "overall_score": 0.0
        }
        
        # Calculate overall quality score
        quality_report["overall_score"] = self._calculate_overall_score(quality_report)
        
        logger.info(f"Quality assessment completed. Overall score: {quality_report['overall_score']:.2f}")
        return quality_report
    
    def _assess_completeness(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess data completeness."""
        completeness = {
            "required_fields": {},
            "optional_fields": {},
            "missing_data_ratio": 0.0
        }
        
        required_fields = ["text", "segment_id", "features"]
        optional_fields = ["source_doc_id", "metadata"]
        
        total_segments = len(segments)
        
        # Check required fields
        for field in required_fields:
            missing_count = sum(1 for seg in segments if not seg.get(field))
            completeness["required_fields"][field] = {
                "present": total_segments - missing_count,
                "missing": missing_count,
                "completeness_ratio": (total_segments - missing_count) / total_segments
            }
        
        # Check optional fields
        for field in optional_fields:
            missing_count = sum(1 for seg in segments if not seg.get(field))
            completeness["optional_fields"][field] = {
                "present": total_segments - missing_count,
                "missing": missing_count,
                "completeness_ratio": (total_segments - missing_count) / total_segments
            }
        
        # Overall missing data ratio
        total_missing = sum(
            field_info["missing"] 
            for field_info in completeness["required_fields"].values()
        )
        total_expected = len(required_fields) * total_segments
        completeness["missing_data_ratio"] = total_missing / total_expected if total_expected > 0 else 0.0
        
        return completeness
    
    def _assess_consistency(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess data consistency."""
        consistency = {
            "text_encoding": self._check_text_encoding(segments),
            "feature_structure": self._check_feature_structure(segments),
            "id_format": self._check_id_format(segments),
            "temporal_context": self._check_temporal_consistency(segments)
        }
        
        return consistency
    
    def _assess_validity(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess data validity."""
        validity = {
            "text_quality": self._check_text_quality(segments),
            "feature_ranges": self._check_feature_ranges(segments),
            "data_types": self._check_data_types(segments)
        }
        
        return validity
    
    def _assess_uniqueness(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess data uniqueness."""
        uniqueness = {
            "duplicate_texts": self._find_duplicate_texts(segments),
            "duplicate_ids": self._find_duplicate_ids(segments),
            "near_duplicates": self._find_near_duplicates(segments)
        }
        
        return uniqueness
    
    def _detect_outliers(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect outliers in the data."""
        outliers = {
            "text_length_outliers": self._find_text_length_outliers(segments),
            "feature_outliers": self._find_feature_outliers(segments)
        }
        
        return outliers
    
    def _check_text_encoding(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check text encoding consistency."""
        encoding_issues = 0
        total_texts = 0
        
        for segment in segments:
            text = segment.get("text", "")
            if text:
                total_texts += 1
                # Check for encoding issues
                try:
                    text.encode('utf-8').decode('utf-8')
                except UnicodeError:
                    encoding_issues += 1
        
        return {
            "total_texts": total_texts,
            "encoding_issues": encoding_issues,
            "encoding_consistency_ratio": (total_texts - encoding_issues) / total_texts if total_texts > 0 else 1.0
        }
    
    def _check_feature_structure(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check feature structure consistency."""
        feature_structures = []
        
        for segment in segments:
            features = segment.get("features", {})
            if features:
                structure = set(features.keys())
                feature_structures.append(structure)
        
        if not feature_structures:
            return {"consistent": True, "variations": 0}
        
        # Find most common structure
        structure_counts = Counter(frozenset(s) for s in feature_structures)
        most_common_structure = structure_counts.most_common(1)[0][0]
        
        # Count variations
        variations = len(structure_counts) - 1
        consistency_ratio = structure_counts.most_common(1)[0][1] / len(feature_structures)
        
        return {
            "consistent": variations == 0,
            "variations": variations,
            "consistency_ratio": consistency_ratio,
            "most_common_structure": list(most_common_structure)
        }
    
    def _check_id_format(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check ID format consistency."""
        id_patterns = []
        
        for segment in segments:
            segment_id = segment.get("segment_id", "")
            if segment_id:
                # Extract pattern (letters, numbers, underscores, etc.)
                pattern = re.sub(r'\d+', 'N', segment_id)  # Replace numbers with N
                pattern = re.sub(r'[a-zA-Z]+', 'A', pattern)  # Replace letters with A
                id_patterns.append(pattern)
        
        pattern_counts = Counter(id_patterns)
        most_common_pattern = pattern_counts.most_common(1)[0] if pattern_counts else ("", 0)
        
        return {
            "total_ids": len(id_patterns),
            "unique_patterns": len(pattern_counts),
            "most_common_pattern": most_common_pattern[0],
            "pattern_consistency_ratio": most_common_pattern[1] / len(id_patterns) if id_patterns else 0.0
        }
    
    def _check_temporal_consistency(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check temporal context consistency."""
        temporal_contexts = []
        
        for segment in segments:
            features = segment.get("features", {})
            temporal_context = features.get("temporal_context", "unknown")
            temporal_contexts.append(temporal_context)
        
        context_counts = Counter(temporal_contexts)
        valid_contexts = ["present", "future", "past", "unknown"]
        invalid_contexts = [ctx for ctx in context_counts.keys() if ctx not in valid_contexts]
        
        return {
            "total_contexts": len(temporal_contexts),
            "context_distribution": dict(context_counts),
            "invalid_contexts": invalid_contexts,
            "validity_ratio": (len(temporal_contexts) - sum(context_counts[ctx] for ctx in invalid_contexts)) / len(temporal_contexts) if temporal_contexts else 1.0
        }
    
    def _check_text_quality(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check text quality metrics."""
        empty_texts = 0
        very_short_texts = 0  # < 10 characters
        very_long_texts = 0   # > 1000 characters
        special_char_issues = 0
        
        for segment in segments:
            text = segment.get("text", "")
            
            if not text.strip():
                empty_texts += 1
            elif len(text) < 10:
                very_short_texts += 1
            elif len(text) > 1000:
                very_long_texts += 1
            
            # Check for excessive special characters
            special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
            if special_char_ratio > 0.3:  # More than 30% special characters
                special_char_issues += 1
        
        total_segments = len(segments)
        
        return {
            "empty_texts": empty_texts,
            "very_short_texts": very_short_texts,
            "very_long_texts": very_long_texts,
            "special_char_issues": special_char_issues,
            "quality_ratio": (total_segments - empty_texts - special_char_issues) / total_segments if total_segments > 0 else 0.0
        }
    
    def _check_feature_ranges(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if feature values are within expected ranges."""
        range_issues = {
            "word_count": 0,
            "sentence_count": 0,
            "negative_values": 0
        }
        
        for segment in segments:
            features = segment.get("features", {})
            
            word_count = features.get("word_count", 0)
            sentence_count = features.get("sentence_count", 0)
            
            # Check for unrealistic values
            if word_count > 500:  # Very long segment
                range_issues["word_count"] += 1
            
            if sentence_count > 50:  # Too many sentences
                range_issues["sentence_count"] += 1
            
            # Check for negative values
            for key, value in features.items():
                if isinstance(value, (int, float)) and value < 0:
                    range_issues["negative_values"] += 1
                    break
        
        return range_issues
    
    def _check_data_types(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check data type consistency."""
        type_issues = 0
        
        for segment in segments:
            features = segment.get("features", {})
            
            # Check expected types
            expected_types = {
                "word_count": (int, float),
                "sentence_count": (int, float),
                "temporal_context": str,
                "noun_phrases": list,
                "discourse_markers": list
            }
            
            for field, expected_type in expected_types.items():
                if field in features:
                    value = features[field]
                    if not isinstance(value, expected_type):
                        type_issues += 1
        
        return {
            "type_issues": type_issues,
            "type_consistency_ratio": (len(segments) - type_issues) / len(segments) if segments else 1.0
        }
    
    def _find_duplicate_texts(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find duplicate texts."""
        text_counts = Counter()
        
        for segment in segments:
            text = segment.get("text", "").strip().lower()
            if text:
                text_counts[text] += 1
        
        duplicates = {text: count for text, count in text_counts.items() if count > 1}
        
        return {
            "duplicate_count": len(duplicates),
            "total_duplicated_segments": sum(duplicates.values()),
            "uniqueness_ratio": (len(segments) - sum(duplicates.values()) + len(duplicates)) / len(segments) if segments else 1.0
        }
    
    def _find_duplicate_ids(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find duplicate IDs."""
        id_counts = Counter()
        
        for segment in segments:
            segment_id = segment.get("segment_id", "")
            if segment_id:
                id_counts[segment_id] += 1
        
        duplicates = {seg_id: count for seg_id, count in id_counts.items() if count > 1}
        
        return {
            "duplicate_ids": len(duplicates),
            "total_duplicated_segments": sum(duplicates.values()),
            "id_uniqueness_ratio": (len(segments) - sum(duplicates.values()) + len(duplicates)) / len(segments) if segments else 1.0
        }
    
    def _find_near_duplicates(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find near-duplicate texts (simplified implementation)."""
        # Simple implementation based on text length and first/last words
        near_duplicates = 0
        texts = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if text:
                words = text.split()
                if len(words) > 2:
                    signature = (len(words), words[0].lower(), words[-1].lower())
                    texts.append(signature)
        
        signature_counts = Counter(texts)
        near_duplicates = sum(count - 1 for count in signature_counts.values() if count > 1)
        
        return {
            "potential_near_duplicates": near_duplicates,
            "near_duplicate_ratio": near_duplicates / len(segments) if segments else 0.0
        }
    
    def _find_text_length_outliers(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find text length outliers."""
        text_lengths = []
        
        for segment in segments:
            text = segment.get("text", "")
            text_lengths.append(len(text))
        
        if not text_lengths:
            return {"outliers": 0}
        
        # Use IQR method for outlier detection
        q1 = np.percentile(text_lengths, 25)
        q3 = np.percentile(text_lengths, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = sum(1 for length in text_lengths if length < lower_bound or length > upper_bound)
        
        return {
            "outliers": outliers,
            "outlier_ratio": outliers / len(text_lengths),
            "length_stats": {
                "mean": float(np.mean(text_lengths)),
                "median": float(np.median(text_lengths)),
                "std": float(np.std(text_lengths)),
                "q1": float(q1),
                "q3": float(q3)
            }
        }
    
    def _find_feature_outliers(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find feature value outliers."""
        feature_outliers = {}
        
        # Collect numeric features
        numeric_features = ["word_count", "sentence_count", "total_sustainability_score"]
        
        for feature_name in numeric_features:
            values = []
            for segment in segments:
                features = segment.get("features", {})
                value = features.get(feature_name)
                if isinstance(value, (int, float)):
                    values.append(value)
            
            if values:
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = sum(1 for value in values if value < lower_bound or value > upper_bound)
                
                feature_outliers[feature_name] = {
                    "outliers": outliers,
                    "outlier_ratio": outliers / len(values),
                    "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                }
        
        return feature_outliers
    
    def _calculate_overall_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-1)."""
        scores = []
        
        # Completeness score
        completeness = quality_report.get("completeness", {})
        if "missing_data_ratio" in completeness:
            scores.append(1.0 - completeness["missing_data_ratio"])
        
        # Consistency score
        consistency = quality_report.get("consistency", {})
        feature_structure = consistency.get("feature_structure", {})
        if "consistency_ratio" in feature_structure:
            scores.append(feature_structure["consistency_ratio"])
        
        # Validity score
        validity = quality_report.get("validity", {})
        text_quality = validity.get("text_quality", {})
        if "quality_ratio" in text_quality:
            scores.append(text_quality["quality_ratio"])
        
        # Uniqueness score
        uniqueness = quality_report.get("uniqueness", {})
        duplicate_texts = uniqueness.get("duplicate_texts", {})
        if "uniqueness_ratio" in duplicate_texts:
            scores.append(duplicate_texts["uniqueness_ratio"])
        
        return float(np.mean(scores)) if scores else 0.0
