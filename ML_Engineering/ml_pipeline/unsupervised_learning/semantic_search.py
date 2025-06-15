"""
Semantic Search module using FAISS for French sustainability text analysis.

This module implements efficient semantic search capabilities using
sentence embeddings and FAISS for similarity search.
"""

import logging
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    SentenceTransformer = None

# Configure logging
logger = logging.getLogger(__name__)

class SemanticSearch:
    """
    Semantic search engine using sentence embeddings and FAISS.
    """
    
    def __init__(self, 
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 index_type: str = "flat"):
        """
        Initialize the semantic search engine.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS and dependencies not available. Semantic search will be limited.")
            self.embedding_model = None
            self.index = None
            self.documents = []
            self.document_metadata = []
            self.embeddings = None
            return
        
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        
        # Initialize components
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.document_metadata = []
        self.embeddings = None
        
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence embedding model."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def _create_index(self, dimension: int, num_documents: int) -> Any:
        """
        Create FAISS index based on configuration.
        
        Args:
            dimension: Embedding dimension
            num_documents: Number of documents
            
        Returns:
            FAISS index
        """
        if self.index_type == "flat":
            # Simple flat index for exact search
            index = faiss.IndexFlatL2(dimension)
        elif self.index_type == "ivf":
            # IVF index for faster approximate search
            nlist = min(100, max(1, num_documents // 10))  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif self.index_type == "hnsw":
            # HNSW index for very fast approximate search
            index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            logger.warning(f"Unknown index type {self.index_type}, using flat index")
            index = faiss.IndexFlatL2(dimension)
        
        return index
    
    def build_index(self, segments: List[Dict[str, Any]]) -> bool:
        """
        Build search index from text segments.

        Args:
            segments: List of text segments with metadata

        Returns:
            True if successful, False otherwise
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - cannot build search index")
            return False

        if not segments:
            logger.warning("No segments provided for indexing")
            return False
        
        try:
            # Extract texts and metadata
            texts = []
            metadata = []
            
            for i, segment in enumerate(segments):
                text = segment.get("text", "")
                if isinstance(text, list):
                    text = " ".join(text)
                
                if text.strip():
                    texts.append(text)
                    metadata.append({
                        "segment_id": segment.get("segment_id", f"segment_{i}"),
                        "source_doc_id": segment.get("source_doc_id", "unknown"),
                        "segment_index": segment.get("segment_index", i),
                        "features": segment.get("features", {}),
                        "original_segment": segment
                    })
            
            if not texts:
                logger.warning("No valid texts found for indexing")
                return False
            
            logger.info(f"Creating embeddings for {len(texts)} documents...")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = self._create_index(dimension, len(texts))
            
            # Train index if needed (for IVF)
            if hasattr(self.index, 'train'):
                logger.info("Training FAISS index...")
                self.index.train(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            # Store data
            self.documents = texts
            self.document_metadata = metadata
            self.embeddings = embeddings
            
            logger.info(f"Built search index with {self.index.ntotal} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error building search index: {e}")
            return False
    
    def search(self, query: str, k: int = 5, 
               filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of search results with scores and metadata
        """
        if self.index is None or not self.documents:
            logger.error("Search index not built. Call build_index() first.")
            return []
        
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in index
            distances, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.document_metadata):
                    metadata = self.document_metadata[idx]
                    
                    # Apply filters if provided
                    if filter_criteria and not self._matches_filter(metadata, filter_criteria):
                        continue
                    
                    result = {
                        "rank": i + 1,
                        "score": float(distance),
                        "similarity": 1.0 / (1.0 + distance),  # Convert distance to similarity
                        "text": self.documents[idx],
                        "segment_id": metadata["segment_id"],
                        "source_doc_id": metadata["source_doc_id"],
                        "segment_index": metadata["segment_index"],
                        "features": metadata["features"],
                        "metadata": metadata
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], 
                       filter_criteria: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filter criteria.
        
        Args:
            metadata: Document metadata
            filter_criteria: Filter criteria
            
        Returns:
            True if matches, False otherwise
        """
        try:
            for key, value in filter_criteria.items():
                if key == "temporal_context":
                    features = metadata.get("features", {})
                    if features.get("temporal_context") != value:
                        return False
                elif key == "source_doc_id":
                    if metadata.get("source_doc_id") != value:
                        return False
                elif key == "min_word_count":
                    features = metadata.get("features", {})
                    if features.get("word_count", 0) < value:
                        return False
                # Add more filter criteria as needed
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return True  # Default to include if filter fails
    
    def get_similar_segments(self, segment_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find segments similar to a given segment.
        
        Args:
            segment_id: ID of the reference segment
            k: Number of similar segments to return
            
        Returns:
            List of similar segments
        """
        # Find the reference segment
        ref_idx = None
        for i, metadata in enumerate(self.document_metadata):
            if metadata["segment_id"] == segment_id:
                ref_idx = i
                break
        
        if ref_idx is None:
            logger.error(f"Segment {segment_id} not found in index")
            return []
        
        # Use the segment text as query
        ref_text = self.documents[ref_idx]
        return self.search(ref_text, k + 1)[1:]  # Exclude the reference segment itself
    
    def save_index(self, filepath: str) -> bool:
        """
        Save the search index and metadata.
        
        Args:
            filepath: Path to save the index (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            logger.error("No index to save")
            return False
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            metadata = {
                "documents": self.documents,
                "document_metadata": self.document_metadata,
                "embedding_model_name": self.embedding_model_name,
                "index_type": self.index_type
            }
            
            with open(f"{filepath}_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Search index saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving search index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load a pre-built search index.
        
        Args:
            filepath: Path to the saved index (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            
            self.documents = metadata["documents"]
            self.document_metadata = metadata["document_metadata"]
            
            # Verify embedding model compatibility
            if metadata["embedding_model_name"] != self.embedding_model_name:
                logger.warning(f"Loaded index uses different embedding model: {metadata['embedding_model_name']}")
            
            logger.info(f"Search index loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading search index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {}
        
        return {
            "total_documents": self.index.ntotal,
            "embedding_dimension": self.index.d if hasattr(self.index, 'd') else None,
            "index_type": self.index_type,
            "embedding_model": self.embedding_model_name,
            "is_trained": getattr(self.index, 'is_trained', True)
        }
