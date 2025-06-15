"""
Metrics module for evaluating ML models and unsupervised learning results.

This module provides comprehensive evaluation metrics for topic modeling,
semantic search, and other ML components.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math

try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    silhouette_score = None
    calinski_harabasz_score = None
    cosine_similarity = None

# Configure logging
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Comprehensive metrics calculator for ML evaluation.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Some metrics will be unavailable.")
    
    def calculate_topic_coherence(self, topics: List[List[str]], 
                                 documents: List[str]) -> Dict[str, float]:
        """
        Calculate topic coherence metrics.
        
        Args:
            topics: List of topics, each containing list of keywords
            documents: List of documents used for topic modeling
            
        Returns:
            Dictionary with coherence metrics
        """
        if not topics or not documents:
            return {"coherence_score": 0.0, "avg_topic_coherence": 0.0}
        
        try:
            # Calculate PMI-based coherence
            coherence_scores = []
            
            for topic_words in topics:
                if len(topic_words) < 2:
                    coherence_scores.append(0.0)
                    continue
                
                topic_coherence = self._calculate_pmi_coherence(topic_words, documents)
                coherence_scores.append(topic_coherence)
            
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
            
            return {
                "coherence_score": float(avg_coherence),
                "avg_topic_coherence": float(avg_coherence),
                "topic_coherences": coherence_scores,
                "num_topics": len(topics)
            }
            
        except Exception as e:
            logger.error(f"Error calculating topic coherence: {e}")
            return {"coherence_score": 0.0, "avg_topic_coherence": 0.0}
    
    def _calculate_pmi_coherence(self, words: List[str], documents: List[str]) -> float:
        """Calculate PMI-based coherence for a topic."""
        if len(words) < 2:
            return 0.0
        
        # Convert documents to lowercase for matching
        docs_lower = [doc.lower() for doc in documents]
        total_docs = len(docs_lower)
        
        # Calculate word frequencies
        word_doc_freq = {}
        for word in words:
            word_doc_freq[word] = sum(1 for doc in docs_lower if word.lower() in doc)
        
        # Calculate PMI for word pairs
        pmi_scores = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word1, word2 = words[i], words[j]
                
                # Count co-occurrences
                cooccur = sum(1 for doc in docs_lower 
                             if word1.lower() in doc and word2.lower() in doc)
                
                if cooccur > 0:
                    # PMI calculation
                    p_word1 = word_doc_freq[word1] / total_docs
                    p_word2 = word_doc_freq[word2] / total_docs
                    p_cooccur = cooccur / total_docs
                    
                    pmi = math.log(p_cooccur / (p_word1 * p_word2))
                    pmi_scores.append(pmi)
        
        return np.mean(pmi_scores) if pmi_scores else 0.0
    
    def calculate_search_metrics(self, search_results: List[Dict[str, Any]], 
                                ground_truth: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate search quality metrics.
        
        Args:
            search_results: List of search results with scores
            ground_truth: Optional list of relevant document IDs
            
        Returns:
            Dictionary with search metrics
        """
        if not search_results:
            return {"avg_score": 0.0, "score_variance": 0.0}
        
        # Basic score statistics
        scores = [result.get("score", 0.0) for result in search_results]
        similarities = [result.get("similarity", 0.0) for result in search_results]
        
        metrics = {
            "avg_score": float(np.mean(scores)),
            "score_variance": float(np.var(scores)),
            "avg_similarity": float(np.mean(similarities)),
            "similarity_variance": float(np.var(similarities)),
            "num_results": len(search_results)
        }
        
        # If ground truth is available, calculate precision/recall
        if ground_truth:
            retrieved_ids = [result.get("segment_id", "") for result in search_results]
            
            # Precision and recall
            relevant_retrieved = len(set(retrieved_ids) & set(ground_truth))
            precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
            recall = relevant_retrieved / len(ground_truth) if ground_truth else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics.update({
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "relevant_retrieved": relevant_retrieved
            })
        
        return metrics
    
    def calculate_clustering_metrics(self, embeddings: np.ndarray, 
                                   labels: List[int]) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.
        
        Args:
            embeddings: Document embeddings
            labels: Cluster labels
            
        Returns:
            Dictionary with clustering metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for clustering metrics")
            return {}
        
        if len(embeddings) == 0 or len(set(labels)) < 2:
            return {"silhouette_score": 0.0, "calinski_harabasz_score": 0.0}
        
        try:
            # Silhouette score
            silhouette = silhouette_score(embeddings, labels)
            
            # Calinski-Harabasz score
            calinski = calinski_harabasz_score(embeddings, labels)
            
            # Additional metrics
            num_clusters = len(set(labels))
            cluster_sizes = Counter(labels)
            avg_cluster_size = np.mean(list(cluster_sizes.values()))
            cluster_size_variance = np.var(list(cluster_sizes.values()))
            
            return {
                "silhouette_score": float(silhouette),
                "calinski_harabasz_score": float(calinski),
                "num_clusters": num_clusters,
                "avg_cluster_size": float(avg_cluster_size),
                "cluster_size_variance": float(cluster_size_variance)
            }
            
        except Exception as e:
            logger.error(f"Error calculating clustering metrics: {e}")
            return {"silhouette_score": 0.0, "calinski_harabasz_score": 0.0}
    
    def calculate_feature_quality_metrics(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics for feature quality assessment.
        
        Args:
            segments: List of segments with features
            
        Returns:
            Dictionary with feature quality metrics
        """
        if not segments:
            return {}
        
        metrics = {
            "total_segments": len(segments),
            "feature_coverage": {},
            "feature_statistics": {},
            "data_quality": {}
        }
        
        # Analyze feature coverage
        feature_fields = ["features", "topics", "sentiment", "paradox", "temporal_context"]
        
        for field in feature_fields:
            count = sum(1 for seg in segments if seg.get(field) is not None)
            metrics["feature_coverage"][field] = count / len(segments)
        
        # Analyze specific features
        if any(seg.get("features") for seg in segments):
            self._analyze_feature_statistics(segments, metrics)
        
        # Data quality checks
        self._analyze_data_quality(segments, metrics)
        
        return metrics
    
    def _analyze_feature_statistics(self, segments: List[Dict[str, Any]], 
                                  metrics: Dict[str, Any]):
        """Analyze statistics of extracted features."""
        feature_stats = {}
        
        # Collect feature values
        word_counts = []
        sentence_counts = []
        sustainability_scores = []
        temporal_confidences = []
        
        for segment in segments:
            features = segment.get("features", {})
            
            # Basic counts
            if "word_count" in features:
                word_counts.append(features["word_count"])
            if "sentence_count" in features:
                sentence_counts.append(features["sentence_count"])
            
            # Sustainability scores
            if "total_sustainability_score" in features:
                sustainability_scores.append(features["total_sustainability_score"])
            
            # Temporal confidence
            if "temporal_confidence" in features:
                temp_conf = features["temporal_confidence"]
                if isinstance(temp_conf, dict):
                    max_conf = max(temp_conf.values()) if temp_conf.values() else 0.0
                    temporal_confidences.append(max_conf)
        
        # Calculate statistics
        if word_counts:
            feature_stats["word_count"] = {
                "mean": float(np.mean(word_counts)),
                "std": float(np.std(word_counts)),
                "min": int(min(word_counts)),
                "max": int(max(word_counts))
            }
        
        if sentence_counts:
            feature_stats["sentence_count"] = {
                "mean": float(np.mean(sentence_counts)),
                "std": float(np.std(sentence_counts)),
                "min": int(min(sentence_counts)),
                "max": int(max(sentence_counts))
            }
        
        if sustainability_scores:
            feature_stats["sustainability_score"] = {
                "mean": float(np.mean(sustainability_scores)),
                "std": float(np.std(sustainability_scores)),
                "min": float(min(sustainability_scores)),
                "max": float(max(sustainability_scores))
            }
        
        if temporal_confidences:
            feature_stats["temporal_confidence"] = {
                "mean": float(np.mean(temporal_confidences)),
                "std": float(np.std(temporal_confidences)),
                "min": float(min(temporal_confidences)),
                "max": float(max(temporal_confidences))
            }
        
        metrics["feature_statistics"] = feature_stats
    
    def _analyze_data_quality(self, segments: List[Dict[str, Any]], 
                            metrics: Dict[str, Any]):
        """Analyze data quality metrics."""
        quality_metrics = {}
        
        # Check for empty texts
        empty_texts = sum(1 for seg in segments if not seg.get("text", "").strip())
        quality_metrics["empty_text_ratio"] = empty_texts / len(segments)
        
        # Check for missing IDs
        missing_ids = sum(1 for seg in segments if not seg.get("segment_id"))
        quality_metrics["missing_id_ratio"] = missing_ids / len(segments)
        
        # Check for duplicate texts
        texts = [seg.get("text", "") for seg in segments]
        unique_texts = len(set(texts))
        quality_metrics["text_uniqueness_ratio"] = unique_texts / len(texts)
        
        # Check feature completeness
        complete_features = 0
        for segment in segments:
            features = segment.get("features", {})
            if (features.get("word_count", 0) > 0 and 
                features.get("sentence_count", 0) > 0 and
                features.get("temporal_context") != "unknown"):
                complete_features += 1
        
        quality_metrics["feature_completeness_ratio"] = complete_features / len(segments)
        
        metrics["data_quality"] = quality_metrics
    
    def generate_evaluation_report(self, segments: List[Dict[str, Any]], 
                                 topic_results: Optional[Dict[str, Any]] = None,
                                 search_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            segments: Processed segments
            topic_results: Topic modeling results
            search_results: Search results for evaluation
            
        Returns:
            Comprehensive evaluation report
        """
        from datetime import datetime
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "detailed_metrics": {}
        }
        
        # Feature quality metrics
        feature_metrics = self.calculate_feature_quality_metrics(segments)
        report["detailed_metrics"]["feature_quality"] = feature_metrics
        
        # Topic modeling metrics
        if topic_results:
            topics = topic_results.get("topics", [])
            documents = [seg.get("text", "") for seg in segments]
            
            if topics and documents:
                topic_keywords = []
                for topic_info in topics.values():
                    keywords = [kw["term"] for kw in topic_info.get("keywords", [])]
                    topic_keywords.append(keywords)
                
                coherence_metrics = self.calculate_topic_coherence(topic_keywords, documents)
                report["detailed_metrics"]["topic_coherence"] = coherence_metrics
        
        # Search metrics
        if search_results:
            search_metrics = self.calculate_search_metrics(search_results)
            report["detailed_metrics"]["search_quality"] = search_metrics
        
        # Generate summary
        report["summary"] = self._generate_summary(report["detailed_metrics"])
        
        return report
    
    def _generate_summary(self, detailed_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary from detailed metrics."""
        summary = {}
        
        # Feature quality summary
        if "feature_quality" in detailed_metrics:
            fq = detailed_metrics["feature_quality"]
            summary["data_quality_score"] = fq.get("data_quality", {}).get("feature_completeness_ratio", 0.0)
            summary["total_segments"] = fq.get("total_segments", 0)
        
        # Topic coherence summary
        if "topic_coherence" in detailed_metrics:
            tc = detailed_metrics["topic_coherence"]
            summary["topic_coherence_score"] = tc.get("coherence_score", 0.0)
            summary["num_topics"] = tc.get("num_topics", 0)
        
        # Search quality summary
        if "search_quality" in detailed_metrics:
            sq = detailed_metrics["search_quality"]
            summary["search_quality_score"] = sq.get("avg_similarity", 0.0)
        
        # Overall quality score
        scores = [v for k, v in summary.items() if k.endswith("_score")]
        summary["overall_quality_score"] = np.mean(scores) if scores else 0.0
        
        return summary
