"""
Dataset Splitter module for creating train/validation/test splits.

This module implements proper dataset splitting with stratification and
ensures no data leakage between splits.
"""

import logging
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import json
import os

# Configure logging
logger = logging.getLogger(__name__)

class DataSplitter:
    """
    Dataset splitter with stratification and quality controls.
    """
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 validation_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 random_seed: int = 42):
        """
        Initialize the dataset splitter.
        
        Args:
            train_ratio: Proportion for training set
            validation_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility
        """
        # Validate ratios
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        logger.info(f"DataSplitter initialized with ratios: train={train_ratio}, val={validation_ratio}, test={test_ratio}")
    
    def split_segments(self, segments: List[Dict[str, Any]], 
                      stratify_by: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split segments into train/validation/test sets.
        
        Args:
            segments: List of segments to split
            stratify_by: Optional field to stratify by (e.g., 'temporal_context')
            
        Returns:
            Dictionary with 'train', 'validation', and 'test' keys
        """
        if not segments:
            return {"train": [], "validation": [], "test": []}
        
        logger.info(f"Splitting {len(segments)} segments with stratification by: {stratify_by}")
        
        if stratify_by:
            return self._stratified_split(segments, stratify_by)
        else:
            return self._random_split(segments)
    
    def _random_split(self, segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Perform random split without stratification."""
        # Shuffle segments
        shuffled_segments = segments.copy()
        random.shuffle(shuffled_segments)
        
        # Calculate split indices
        n_total = len(shuffled_segments)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.validation_ratio)
        
        # Split
        train_segments = shuffled_segments[:n_train]
        val_segments = shuffled_segments[n_train:n_train + n_val]
        test_segments = shuffled_segments[n_train + n_val:]
        
        logger.info(f"Random split: train={len(train_segments)}, val={len(val_segments)}, test={len(test_segments)}")
        
        return {
            "train": train_segments,
            "validation": val_segments,
            "test": test_segments
        }
    
    def _stratified_split(self, segments: List[Dict[str, Any]], 
                         stratify_by: str) -> Dict[str, List[Dict[str, Any]]]:
        """Perform stratified split to maintain class distribution."""
        # Group segments by stratification key
        groups = defaultdict(list)
        
        for segment in segments:
            # Extract stratification value
            if stratify_by == "temporal_context":
                value = segment.get("features", {}).get("temporal_context", "unknown")
            elif stratify_by == "source_doc_id":
                value = segment.get("source_doc_id", "unknown")
            elif stratify_by == "sustainability_level":
                # Create sustainability level based on score
                score = segment.get("features", {}).get("total_sustainability_score", 0)
                if score == 0:
                    value = "none"
                elif score <= 2:
                    value = "low"
                elif score <= 5:
                    value = "medium"
                else:
                    value = "high"
            else:
                value = segment.get(stratify_by, "unknown")
            
            groups[value].append(segment)
        
        logger.info(f"Stratification groups: {[(k, len(v)) for k, v in groups.items()]}")
        
        # Split each group proportionally
        train_segments = []
        val_segments = []
        test_segments = []
        
        for group_key, group_segments in groups.items():
            if len(group_segments) < 3:
                # If group is too small, put all in training
                logger.warning(f"Group '{group_key}' has only {len(group_segments)} segments, adding all to training")
                train_segments.extend(group_segments)
                continue
            
            # Shuffle group
            shuffled_group = group_segments.copy()
            random.shuffle(shuffled_group)
            
            # Calculate split sizes for this group
            n_group = len(shuffled_group)
            n_train_group = max(1, int(n_group * self.train_ratio))
            n_val_group = max(1, int(n_group * self.validation_ratio))
            
            # Ensure we don't exceed group size
            if n_train_group + n_val_group >= n_group:
                n_val_group = max(0, n_group - n_train_group - 1)
            
            # Split group
            group_train = shuffled_group[:n_train_group]
            group_val = shuffled_group[n_train_group:n_train_group + n_val_group]
            group_test = shuffled_group[n_train_group + n_val_group:]
            
            train_segments.extend(group_train)
            val_segments.extend(group_val)
            test_segments.extend(group_test)
        
        # Final shuffle to mix groups
        random.shuffle(train_segments)
        random.shuffle(val_segments)
        random.shuffle(test_segments)
        
        logger.info(f"Stratified split: train={len(train_segments)}, val={len(val_segments)}, test={len(test_segments)}")
        
        return {
            "train": train_segments,
            "validation": val_segments,
            "test": test_segments
        }
    
    def split_by_document(self, segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split segments ensuring no document appears in multiple splits.
        
        Args:
            segments: List of segments to split
            
        Returns:
            Dictionary with train/validation/test splits
        """
        # Group segments by document
        doc_groups = defaultdict(list)
        for segment in segments:
            doc_id = segment.get("source_doc_id", "unknown")
            doc_groups[doc_id].append(segment)
        
        # Split documents
        doc_ids = list(doc_groups.keys())
        random.shuffle(doc_ids)
        
        n_docs = len(doc_ids)
        n_train_docs = int(n_docs * self.train_ratio)
        n_val_docs = int(n_docs * self.validation_ratio)
        
        train_doc_ids = doc_ids[:n_train_docs]
        val_doc_ids = doc_ids[n_train_docs:n_train_docs + n_val_docs]
        test_doc_ids = doc_ids[n_train_docs + n_val_docs:]
        
        # Collect segments for each split
        train_segments = []
        val_segments = []
        test_segments = []
        
        for doc_id in train_doc_ids:
            train_segments.extend(doc_groups[doc_id])
        
        for doc_id in val_doc_ids:
            val_segments.extend(doc_groups[doc_id])
        
        for doc_id in test_doc_ids:
            test_segments.extend(doc_groups[doc_id])
        
        logger.info(f"Document-based split: train={len(train_segments)} ({len(train_doc_ids)} docs), "
                   f"val={len(val_segments)} ({len(val_doc_ids)} docs), "
                   f"test={len(test_segments)} ({len(test_doc_ids)} docs)")
        
        return {
            "train": train_segments,
            "validation": val_segments,
            "test": test_segments
        }
    
    def validate_split(self, split_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Validate the quality of the split.
        
        Args:
            split_data: Dictionary with train/validation/test splits
            
        Returns:
            Validation report
        """
        train_segments = split_data.get("train", [])
        val_segments = split_data.get("validation", [])
        test_segments = split_data.get("test", [])
        
        total_segments = len(train_segments) + len(val_segments) + len(test_segments)
        
        validation_report = {
            "total_segments": total_segments,
            "split_sizes": {
                "train": len(train_segments),
                "validation": len(val_segments),
                "test": len(test_segments)
            },
            "split_ratios": {
                "train": len(train_segments) / total_segments if total_segments > 0 else 0,
                "validation": len(val_segments) / total_segments if total_segments > 0 else 0,
                "test": len(test_segments) / total_segments if total_segments > 0 else 0
            },
            "data_leakage_check": self._check_data_leakage(split_data),
            "distribution_analysis": self._analyze_distributions(split_data)
        }
        
        return validation_report
    
    def _check_data_leakage(self, split_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Check for data leakage between splits."""
        train_ids = set(seg.get("segment_id", "") for seg in split_data.get("train", []))
        val_ids = set(seg.get("segment_id", "") for seg in split_data.get("validation", []))
        test_ids = set(seg.get("segment_id", "") for seg in split_data.get("test", []))
        
        # Check for overlapping segment IDs
        train_val_overlap = len(train_ids & val_ids)
        train_test_overlap = len(train_ids & test_ids)
        val_test_overlap = len(val_ids & test_ids)
        
        # Check for overlapping documents
        train_docs = set(seg.get("source_doc_id", "") for seg in split_data.get("train", []))
        val_docs = set(seg.get("source_doc_id", "") for seg in split_data.get("validation", []))
        test_docs = set(seg.get("source_doc_id", "") for seg in split_data.get("test", []))
        
        doc_train_val_overlap = len(train_docs & val_docs)
        doc_train_test_overlap = len(train_docs & test_docs)
        doc_val_test_overlap = len(val_docs & test_docs)
        
        leakage_report = {
            "segment_overlaps": {
                "train_validation": train_val_overlap,
                "train_test": train_test_overlap,
                "validation_test": val_test_overlap
            },
            "document_overlaps": {
                "train_validation": doc_train_val_overlap,
                "train_test": doc_train_test_overlap,
                "validation_test": doc_val_test_overlap
            },
            "has_leakage": (train_val_overlap + train_test_overlap + val_test_overlap) > 0
        }
        
        return leakage_report
    
    def _analyze_distributions(self, split_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze feature distributions across splits."""
        distributions = {}
        
        for split_name, segments in split_data.items():
            if not segments:
                continue
            
            # Temporal context distribution
            temporal_contexts = [seg.get("features", {}).get("temporal_context", "unknown") 
                               for seg in segments]
            temporal_dist = Counter(temporal_contexts)
            
            # Document distribution
            doc_ids = [seg.get("source_doc_id", "unknown") for seg in segments]
            doc_dist = Counter(doc_ids)
            
            # Word count statistics
            word_counts = [seg.get("features", {}).get("word_count", 0) for seg in segments]
            
            distributions[split_name] = {
                "temporal_context_distribution": dict(temporal_dist),
                "document_distribution": dict(doc_dist),
                "word_count_stats": {
                    "mean": float(np.mean(word_counts)) if word_counts else 0.0,
                    "std": float(np.std(word_counts)) if word_counts else 0.0,
                    "min": int(min(word_counts)) if word_counts else 0,
                    "max": int(max(word_counts)) if word_counts else 0
                }
            }
        
        return distributions
    
    def save_split(self, split_data: Dict[str, List[Dict[str, Any]]], 
                   output_dir: str, prefix: str = "dataset") -> Dict[str, str]:
        """
        Save split data to files.
        
        Args:
            split_data: Dictionary with train/validation/test splits
            output_dir: Directory to save files
            prefix: Prefix for filenames
            
        Returns:
            Dictionary with paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        for split_name, segments in split_data.items():
            filename = f"{prefix}_{split_name}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Create split metadata
            split_metadata = {
                "split_name": split_name,
                "total_segments": len(segments),
                "random_seed": self.random_seed,
                "split_ratios": {
                    "train": self.train_ratio,
                    "validation": self.validation_ratio,
                    "test": self.test_ratio
                },
                "segments": segments
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(split_metadata, f, ensure_ascii=False, indent=2)
            
            saved_files[split_name] = filepath
            logger.info(f"Saved {split_name} split with {len(segments)} segments to {filepath}")
        
        # Save validation report
        validation_report = self.validate_split(split_data)
        report_path = os.path.join(output_dir, f"{prefix}_split_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        
        saved_files["validation_report"] = report_path
        
        return saved_files
