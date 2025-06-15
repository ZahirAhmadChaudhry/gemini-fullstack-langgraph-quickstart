"""
Topic Modeling module using BERTopic for French sustainability text analysis.

This module implements advanced topic modeling capabilities specifically
adapted for French language and sustainability domain using BERTopic.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    import spacy
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    BERTopic = None
    SentenceTransformer = None
    CountVectorizer = None
    spacy = None

# Configure logging
logger = logging.getLogger(__name__)

class TopicModeling:
    """
    Advanced topic modeling using BERTopic for French sustainability text.
    """
    
    def __init__(self, 
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 language: str = "multilingual",
                 min_topic_size: int = 2,
                 spacy_model: str = "fr_core_news_lg"):
        """
        Initialize the topic modeling system.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            language: Language setting for BERTopic
            min_topic_size: Minimum number of documents per topic
            spacy_model: French spaCy model for preprocessing
        """
        if not BERTOPIC_AVAILABLE:
            logger.warning("BERTopic and dependencies not available. Topic modeling will be limited.")
            self.embedding_model = None
            self.vectorizer_model = None
            self.topic_model = None
            self.nlp = None
            return
        
        self.embedding_model_name = embedding_model
        self.language = language
        self.min_topic_size = min_topic_size
        self.spacy_model = spacy_model
        
        # Initialize components
        self.embedding_model = None
        self.vectorizer_model = None
        self.topic_model = None
        self.nlp = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # Load sentence transformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            
            # Load spaCy model for French stop words
            self.nlp = spacy.load(self.spacy_model)
            french_stop_words = list(self.nlp.Defaults.stop_words)
            
            # Initialize vectorizer with French stop words
            self.vectorizer_model = CountVectorizer(
                stop_words=french_stop_words,
                min_df=1,
                ngram_range=(1, 2)  # Include bigrams for better topic representation
            )
            
            # Initialize BERTopic model
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                vectorizer_model=self.vectorizer_model,
                language=self.language,
                min_topic_size=self.min_topic_size,
                verbose=True
            )
            
            logger.info("Topic modeling components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing topic modeling components: {e}")
            raise
    
    def fit_transform(self, documents: List[str]) -> Tuple[List[int], List[float]]:
        """
        Fit the topic model and transform documents.

        Args:
            documents: List of text documents

        Returns:
            Tuple of (topics, probabilities)
        """
        if not BERTOPIC_AVAILABLE or not self.topic_model:
            logger.warning("BERTopic not available - returning empty results")
            return [], []

        if not documents:
            logger.warning("No documents provided for topic modeling")
            return [], []

        try:
            logger.info(f"Fitting topic model on {len(documents)} documents")
            topics, probs = self.topic_model.fit_transform(documents)

            logger.info(f"Discovered {len(self.topic_model.get_topic_info())} topics")
            return topics.tolist(), probs.tolist()

        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return [], []
    
    def get_topic_info(self) -> Dict[str, Any]:
        """
        Get information about discovered topics.
        
        Returns:
            Dictionary with topic information
        """
        if self.topic_model is None:
            return {}
        
        try:
            topic_info = self.topic_model.get_topic_info()
            
            # Convert to dictionary format
            topics_dict = {}
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                topics_dict[topic_id] = {
                    'count': row['Count'],
                    'name': row.get('Name', f'Topic {topic_id}'),
                    'keywords': self.get_topic_keywords(topic_id)
                }
            
            return topics_dict
            
        except Exception as e:
            logger.error(f"Error getting topic info: {e}")
            return {}
    
    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get keywords for a specific topic.
        
        Args:
            topic_id: Topic identifier
            top_k: Number of top keywords to return
            
        Returns:
            List of keyword dictionaries with term and score
        """
        if self.topic_model is None:
            return []
        
        try:
            topic_words = self.topic_model.get_topic(topic_id)
            if not topic_words:
                return []
            
            keywords = []
            for word, score in topic_words[:top_k]:
                keywords.append({
                    'term': word,
                    'score': float(score),
                    'topic_id': topic_id
                })
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error getting topic keywords: {e}")
            return []
    
    def process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process segments to add topic modeling results.

        Args:
            segments: List of text segments

        Returns:
            Segments with added topic modeling metadata
        """
        if not segments:
            return []

        if not BERTOPIC_AVAILABLE or not self.topic_model:
            logger.warning("BERTopic not available - returning segments without topic modeling")
            return segments
        
        # Extract texts for topic modeling
        documents = []
        valid_indices = []
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "")
            if isinstance(text, list):
                text = " ".join(text)
            
            if text.strip():
                documents.append(text)
                valid_indices.append(i)
        
        if not documents:
            logger.warning("No valid documents found for topic modeling")
            return segments
        
        # Perform topic modeling
        topics, probs = self.fit_transform(documents)
        
        # Add results to segments
        processed_segments = segments.copy()
        
        for doc_idx, segment_idx in enumerate(valid_indices):
            if doc_idx < len(topics):
                topic_id = topics[doc_idx]
                probability = probs[doc_idx] if doc_idx < len(probs) else 0.0
                
                # Get topic keywords
                keywords = self.get_topic_keywords(topic_id, top_k=5)
                
                # Add topic information to segment
                processed_segments[segment_idx]["topic_modeling"] = {
                    "topic_id": topic_id,
                    "probability": probability,
                    "keywords": keywords,
                    "topic_name": f"Topic {topic_id}"
                }
        
        logger.info(f"Added topic modeling results to {len(valid_indices)} segments")
        return processed_segments
    
    def get_topic_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for topic visualization.
        
        Returns:
            Dictionary with visualization data
        """
        if self.topic_model is None:
            return {}
        
        try:
            # Get topic info
            topic_info = self.get_topic_info()
            
            # Prepare visualization data
            viz_data = {
                "topics": topic_info,
                "total_topics": len(topic_info),
                "model_info": {
                    "embedding_model": self.embedding_model_name,
                    "min_topic_size": self.min_topic_size,
                    "language": self.language
                }
            }
            
            return viz_data
            
        except Exception as e:
            logger.error(f"Error preparing visualization data: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained topic model.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.topic_model is None:
            logger.error("No trained model to save")
            return False
        
        try:
            self.topic_model.save(filepath)
            logger.info(f"Topic model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving topic model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a pre-trained topic model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.topic_model = BERTopic.load(filepath)
            logger.info(f"Topic model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading topic model: {e}")
            return False
