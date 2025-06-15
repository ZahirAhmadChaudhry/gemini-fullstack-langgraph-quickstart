"""
Topic identification module using advanced keyword extraction for French sustainability text.

This module implements TextRank, YAKE!, and other keyword extraction techniques
specifically adapted for French language and sustainability domain.
"""

import os
import csv
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
import re
import spacy
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordExtractor:
    """
    Extracts keywords from French text segments using various algorithms.
    """
    
    def __init__(self, 
                 method: str = "textrank",
                 num_keywords: int = 5,
                 sustainability_terms_path: Optional[str] = None,
                 spacy_model: str = "fr_core_news_lg"):
        """
        Initialize the keyword extractor.
        
        Args:
            method: Keyword extraction method ('textrank', 'tfidf', 'rake', 'yake')
            num_keywords: Number of keywords to extract
            sustainability_terms_path: Path to file with sustainability terms
            spacy_model: French spaCy model to use
        """
        self.method = method
        self.num_keywords = num_keywords
        self.sustainability_terms = set()
        self.nlp = None
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise
            
        # Load sustainability terms if provided
        if sustainability_terms_path and os.path.exists(sustainability_terms_path):
            self.load_sustainability_terms(sustainability_terms_path)
    
    def load_sustainability_terms(self, filepath: str) -> None:
        """
        Load sustainability domain terms from file.
        
        Args:
            filepath: Path to file containing sustainability terms
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.csv'):
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # Skip empty rows
                            self.sustainability_terms.add(row[0].lower().strip())
                else:
                    for line in f:
                        term = line.strip().lower()
                        if term:  # Skip empty lines
                            self.sustainability_terms.add(term)
            
            logger.info(f"Loaded {len(self.sustainability_terms)} sustainability terms")
        except Exception as e:
            logger.error(f"Error loading sustainability terms: {e}")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for keyword extraction.
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        if not text or not isinstance(text, str):
            return []
            
        # Process with spaCy
        doc = self.nlp(text)
        
        # Filter tokens: keep nouns, adjectives, verbs (except auxiliaries)
        # Use lemmas and convert to lowercase
        tokens = [token.lemma_.lower() for token in doc 
                 if (token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']) 
                 and not token.is_stop
                 and not token.is_punct
                 and len(token.text) > 1]
        
        return tokens
    
    def textrank(self, tokens: List[str], window_size: int = 4) -> List[Tuple[str, float]]:
        """
        Extract keywords using TextRank algorithm.
        
        Args:
            tokens: Preprocessed tokens
            window_size: Window size for co-occurrence
            
        Returns:
            List of (keyword, score) tuples
        """
        if not tokens:
            return []
            
        # Build co-occurrence graph
        graph = nx.Graph()
        
        # Add nodes
        for token in set(tokens):
            graph.add_node(token)
        
        # Add edges based on co-occurrence within window
        for i, token in enumerate(tokens):
            for j in range(i+1, min(i+window_size, len(tokens))):
                if token != tokens[j]:
                    if graph.has_edge(token, tokens[j]):
                        graph[token][tokens[j]]['weight'] += 1.0
                    else:
                        graph.add_edge(token, tokens[j], weight=1.0)
        
        # Apply PageRank algorithm
        try:
            scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
        except:
            # If PageRank fails (e.g., with disconnected graph), use degree centrality
            logger.warning("PageRank failed, using degree centrality instead")
            scores = nx.degree_centrality(graph)
            
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_scores
    
    def tfidf(self, text: str, corpus: List[str]) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF scores.
        
        Args:
            text: Current text segment
            corpus: All text segments for IDF calculation
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text or not corpus:
            return []
            
        # Initialize vectorizer
        vectorizer = TfidfVectorizer(
            min_df=1, 
            norm='l2',
            smooth_idf=True,
            use_idf=True,
            ngram_range=(1, 1),
            stop_words=self.nlp.Defaults.stop_words
        )
        
        # Fit on corpus
        vectorizer.fit(corpus)
        
        # Transform current document
        tfidf_matrix = vectorizer.transform([text])
        
        # Get scores for each word
        feature_names = vectorizer.get_feature_names_out()
        dense = tfidf_matrix.todense()
        scores = dense[0].tolist()[0]
        
        # Create word-score pairs
        word_scores = [(feature_names[i], score) for i, score in enumerate(scores) if score > 0]
        
        # Sort by score
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores
    
    def yake(self, text: str, max_ngram_size: int = 3, window_size: int = 2) -> List[Tuple[str, float]]:
        """
        Extract keywords using YAKE! algorithm (Yet Another Keyword Extractor).
        
        YAKE! uses a set of statistical features extracted from single documents to select
        the most relevant keywords from texts without need for a corpus or external resources.
        Lower scores indicate more relevant keywords (unlike TextRank and TF-IDF).
        
        Args:
            text: Input text string
            max_ngram_size: Maximum n-gram size for keywords
            window_size: Window size for co-occurrence
            
        Returns:
            List of (keyword, score) tuples, sorted by relevance (lower score = more relevant)
        """
        if not text:
            return []
            
        # Process with spaCy to get tokens, sentences and keep track of positions
        doc = self.nlp(text)
        
        # Get sentences for statistical features calculation
        sentences = [sent.text for sent in doc.sents]
        
        # Calculate term frequency
        term_freq = Counter()
        
        # Process tokens and collect features
        tokens = []
        for token in doc:
            if (token.is_alpha and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 1):
                # Convert to lowercase for consistency
                word = token.text.lower()
                term_freq[word] += 1
                tokens.append(word)
        
        if not tokens:
            return []
            
        # Calculate term features
        word_scores = {}
        
        for word in term_freq:
            # Feature 1: Term Frequency (TF)
            # Normalized between 0 and 1
            tf = term_freq[word] / len(tokens)
            
            # Feature 2: Term Positions
            # Calculate mean position of word occurrences (first position = 1)
            positions = [i+1 for i, t in enumerate(tokens) if t == word]
            mean_position = sum(positions) / len(positions)
            
            # Feature 3: Term's contexts diversity
            # Count of different terms that appear around this term within window
            contexts = set()
            for i, t in enumerate(tokens):
                if t == word:
                    # Collect contexts (surrounding words within window)
                    context_start = max(0, i - window_size)
                    context_end = min(len(tokens), i + window_size + 1)
                    for j in range(context_start, context_end):
                        if j != i:  # Exclude the word itself
                            contexts.add(tokens[j])
            
            # Feature 4: Sentence spread
            # Count of different sentences the term appears in
            sent_spread = sum(1 for sent in sentences if word in sent.lower())
            
            # Calculate the YAKE! score (lower is better)
            # Formula based on YAKE! paper with slight simplification
            # Score = (TF * (1 + Position)) / (Context_Diversity * Sentence_Spread)
            # For this implementation, we want higher diversity and sentence spread to lower the score
            
            # Normalize features for better behavior
            norm_tf = tf  # Already normalized
            norm_position = 1.0 - (1.0 / (1.0 + mean_position))  # Convert to [0,1], higher position = higher value
            norm_context = len(contexts) / (2 * window_size)  # Normalize by max possible contexts
            # Ensure non-zero denominator
            if norm_context == 0:
                norm_context = 0.1
            
            # Ensure non-zero sentence spread
            if sent_spread == 0:
                sent_spread = 1
                
            # YAKE! score calculation (lower is better)
            score = (norm_tf * (1 + norm_position)) / (norm_context * sent_spread)
            
            word_scores[word] = score
            
        # Generate candidate n-grams and score them
        candidates = []
        
        # For unigrams, use the calculated scores
        for word, score in word_scores.items():
            candidates.append((word, score))
            
        # For n-grams (n > 1), calculate score based on constituent word scores
        if max_ngram_size > 1:
            # Extract candidate n-grams from the document
            doc = self.nlp(text)
            for n in range(2, min(max_ngram_size + 1, 4)):  # Up to max_ngram_size, with a hard limit of 3
                for i in range(len(doc) - n + 1):
                    # Check if the span forms a valid n-gram (no stop words, punctuation, etc.)
                    span = doc[i:i+n]
                    # Basic filtering: skip if contains stops, punctuation or has invalid POS
                    if any(token.is_stop or token.is_punct for token in span):
                        continue
                    
                    # Check that it's a valid phrase (most tokens are NOUN, ADJ, VERB, PROPN)
                    valid_pos = ['NOUN', 'ADJ', 'VERB', 'PROPN']
                    if sum(1 for token in span if token.pos_ in valid_pos) < (n // 2 + 1):
                        continue
                    
                    # Get the n-gram text
                    ngram = span.text.lower()
                    
                    # Calculate score for n-gram as product of constituent word scores
                    # divided by the length to penalize longer n-grams slightly
                    ngram_tokens = [token.text.lower() for token in span 
                                   if not token.is_stop and not token.is_punct]
                    
                    if not ngram_tokens:
                        continue
                    
                    # Get scores for constituent words, use a default high score if not found
                    constituent_scores = [word_scores.get(token, 1.0) for token in ngram_tokens]
                    
                    # Calculate n-gram score (lower is better in YAKE!)
                    if len(constituent_scores) > 0:
                        ngram_score = sum(constituent_scores) / len(constituent_scores)
                        candidates.append((ngram, ngram_score))
        
        # Sort by score (lower is better in YAKE!) and remove duplicates
        seen = set()
        unique_candidates = []
        for candidate, score in sorted(candidates, key=lambda x: x[1]):
            # Simple deduplication - skip if we've seen this candidate
            if candidate in seen:
                continue
                
            seen.add(candidate)
            unique_candidates.append((candidate, score))
        
        return unique_candidates
    
    def extract_keywords(self, 
                        text: str, 
                        method: Optional[str] = None,
                        corpus: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract keywords from a text segment.
        
        Args:
            text: Text segment to analyze
            method: Override default method if provided
            corpus: All text segments (needed for some methods)
            
        Returns:
            List of keyword objects with term and score
        """
        if not text:
            return []
            
        method = method or self.method
        
        # Preprocess text
        tokens = self.preprocess_text(text)
        if not tokens and method != "yake":  # YAKE does its own preprocessing
            return []
        
        # Extract keywords using the specified method
        if method == "textrank":
            keyword_scores = self.textrank(tokens)
        elif method == "tfidf":
            if not corpus:
                logger.warning("Corpus required for TF-IDF, falling back to TextRank")
                keyword_scores = self.textrank(tokens)
            else:
                keyword_scores = self.tfidf(text, corpus)
        elif method == "yake":
            keyword_scores = self.yake(text)
            # Invert YAKE scores for consistency (higher = better)
            if keyword_scores:
                max_score = max(score for _, score in keyword_scores)
                keyword_scores = [(term, max_score - score + 0.01) for term, score in keyword_scores]
        else:
            logger.warning(f"Unsupported method: {method}, falling back to TextRank")
            keyword_scores = self.textrank(tokens)
            
        # Format results, boost sustainability terms if available
        keywords = []
        for term, score in keyword_scores:
            # Boost score for sustainability domain terms
            if self.sustainability_terms and term.lower() in self.sustainability_terms:
                score *= 1.5
                
            keywords.append({
                "term": term,
                "score": float(score),
                "is_sustainability_term": term.lower() in self.sustainability_terms
            })
        
        # Return top keywords
        return sorted(keywords, key=lambda x: x["score"], reverse=True)[:self.num_keywords]
    
    def process_segment(self, segment: Dict[str, Any], corpus: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a single text segment to identify topics.
        
        Args:
            segment: Text segment to analyze
            corpus: All text segments (needed for some methods)
            
        Returns:
            Segment with added topic metadata
        """
        # Extract text from segment
        text = segment.get("text", "")
        if isinstance(text, list):
            # Join if text is a list of sentences
            text = " ".join(text)
            
        if not text:
            logger.warning(f"Empty text in segment {segment.get('segment_id', 'unknown')}")
            segment["topics"] = []
            return segment
            
        # Extract keywords
        keywords = self.extract_keywords(text, corpus=corpus)
        
        # Add topics to segment
        segment["topics"] = keywords
        
        return segment
    
    def process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple text segments to identify topics.
        
        Args:
            segments: List of text segments to analyze
            
        Returns:
            Segments with added topic metadata
        """
        if not segments:
            return []
            
        # For TF-IDF, prepare corpus
        corpus = None
        if self.method == "tfidf":
            corpus = []
            for segment in segments:
                text = segment.get("text", "")
                if isinstance(text, list):
                    text = " ".join(text)
                corpus.append(text)
        
        # Process each segment
        processed_segments = []
        for segment in segments:
            processed = self.process_segment(segment, corpus)
            processed_segments.append(processed)
            
        logger.info(f"Processed {len(processed_segments)} segments for topic identification")
        return processed_segments

