"""
Opinion detection module using sentiment analysis for French sustainability discussions.

This module implements both lexicon-based and transformer-based sentiment analysis with
French-specific adaptations including negation handling and contrastive marker detection.
"""

import os
import csv
import re
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import transformer components conditionally to handle cases where they're not installed
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not available. Transformer-based sentiment analysis will not work.")
    TRANSFORMERS_AVAILABLE = False

class SentimentAnalyzer:
    """
    Performs sentiment analysis on French text segments.
    """
    
    def __init__(self, 
                 method: str = "lexicon_based",
                 lexicon_name: str = "FEEL",
                 lexicon_path: Optional[str] = None,
                 negation_handling: bool = True,
                 transformer_model: str = "nlptown/bert-base-multilingual-uncased-sentiment",
                 spacy_model: str = "fr_core_news_lg"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            method: Sentiment analysis method ('lexicon_based', 'transformer_based')
            lexicon_name: Name of lexicon to use ('FEEL', 'LSDfr', 'UniSent')
            lexicon_path: Path to sentiment lexicon file
            negation_handling: Whether to handle negations in text
            transformer_model: Hugging Face model ID for transformer-based analysis
            spacy_model: French spaCy model to use
        """
        self.method = method
        self.lexicon_name = lexicon_name
        self.negation_handling = negation_handling
        self.transformer_model = transformer_model
        self.sentiment_lexicon = {}
        self.nlp = None
        self.transformer_pipeline = None
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise
            
        # Load sentiment lexicon if provided
        if lexicon_path and os.path.exists(lexicon_path):
            self.load_sentiment_lexicon(lexicon_path)
        else:
            if method == "lexicon_based":
                logger.warning("No sentiment lexicon provided or file not found. Lexicon-based analysis will be limited.")
        
        # Load transformer model if specified
        if method == "transformer_based":
            if TRANSFORMERS_AVAILABLE:
                try:
                    self._load_transformer_model(transformer_model)
                except Exception as e:
                    logger.error(f"Error loading transformer model: {e}")
                    logger.warning("Falling back to lexicon-based analysis")
                    self.method = "lexicon_based"
            else:
                logger.warning("Transformers library not available. Falling back to lexicon-based analysis.")
                self.method = "lexicon_based"
    
    def _load_transformer_model(self, model_name: str) -> None:
        """
        Load a transformer model for sentiment analysis.
        
        Args:
            model_name: Hugging Face model identifier
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
            
        logger.info(f"Loading transformer model: {model_name}")
        
        try:
            # For French sentiment analysis, use a PyTorch model instead of TensorFlow
            # Options:
            # - "camembert-base-sentiment" - CamemBERT model fine-tuned for sentiment
            # - "nlptown/bert-base-multilingual-uncased-sentiment" - Multilingual model
            # - "cmarkea/distilcamembert-base-sentiment" - Smaller, faster model
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True
            )
            logger.info(f"Successfully loaded transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            raise
    
    def load_sentiment_lexicon(self, filepath: str) -> None:
        """
        Load sentiment lexicon from file.
        
        Args:
            filepath: Path to sentiment lexicon file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.csv'):
                    reader = csv.reader(f)
                    header = next(reader, None)
                    
                    # Determine column indices based on lexicon format
                    if self.lexicon_name == "FEEL":
                        # FEEL format: term,polarity
                        term_idx, polarity_idx = 0, 1
                    else:
                        # Default format: assume term in first column, score in second
                        term_idx, polarity_idx = 0, 1
                    
                    for row in reader:
                        if len(row) > polarity_idx:
                            term = row[term_idx].lower().strip()
                            
                            # Convert polarity to numeric score
                            polarity = row[polarity_idx].strip()
                            if polarity in ['positive', 'pos']:
                                score = 1.0
                            elif polarity in ['negative', 'neg']:
                                score = -1.0
                            elif polarity in ['neutral', 'neu']:
                                score = 0.0
                            else:
                                try:
                                    score = float(polarity)
                                except ValueError:
                                    logger.warning(f"Could not parse polarity value: {polarity}")
                                    continue
                            
                            self.sentiment_lexicon[term] = score
                else:
                    # Simple format: term<tab>score
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            term = parts[0].lower()
                            try:
                                score = float(parts[1])
                                self.sentiment_lexicon[term] = score
                            except ValueError:
                                continue
            
            logger.info(f"Loaded {len(self.sentiment_lexicon)} terms in sentiment lexicon")
        except Exception as e:
            logger.error(f"Error loading sentiment lexicon: {e}")
    
    def detect_negations(self, doc: spacy.tokens.Doc) -> List[Tuple[int, int]]:
        """
        Detect negation spans in a spaCy document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of (start_idx, end_idx) tuples for negation spans
        """
        negation_cues = {"ne", "n'", "pas", "jamais", "aucun", "aucune", "rien", "personne", "ni", "non"}
        negation_spans = []
        
        for i, token in enumerate(doc):
            if token.text.lower() in negation_cues or token.lemma_.lower() in negation_cues:
                # French negation often has two parts (ne...pas)
                # Look for "ne" or "n'" preceding "pas", "jamais", etc.
                if token.text.lower() in {"ne", "n'"}:
                    # Find the second part within the next 4 tokens
                    for j in range(i+1, min(i+5, len(doc))):
                        if doc[j].text.lower() in {"pas", "jamais", "plus", "guère", "point"}:
                            # The negation affects typically up to 3 tokens after the second part
                            end_idx = min(j+4, len(doc))
                            negation_spans.append((i, end_idx))
                            break
                else:
                    # Single negation words like "jamais", "aucun", "rien"
                    # Typically affect the next 3 tokens
                    end_idx = min(i+4, len(doc))
                    negation_spans.append((i, end_idx))
        
        return negation_spans
    
    def detect_contrastive_markers(self, doc: spacy.tokens.Doc) -> List[int]:
        """
        Detect contrastive markers in a spaCy document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of token indices for contrastive markers
        """
        # Common French contrastive markers
        contrastive_markers = {
            "mais", "cependant", "toutefois", "néanmoins", "pourtant", 
            "en revanche", "au contraire", "or", "tandis que", "alors que",
            "bien que", "malgré", "en dépit de", "quoique", "même si",
            "par contre", "d'un côté", "d'autre part", "d'une part"
        }
        
        marker_indices = []
        
        for i, token in enumerate(doc):
            # Single word markers
            if token.text.lower() in contrastive_markers:
                marker_indices.append(i)
            
            # Multi-word markers (check bigrams and trigrams)
            if i < len(doc) - 1:
                bigram = token.text.lower() + " " + doc[i+1].text.lower()
                if bigram in contrastive_markers:
                    marker_indices.append(i)
            
            if i < len(doc) - 2:
                trigram = token.text.lower() + " " + doc[i+1].text.lower() + " " + doc[i+2].text.lower()
                if trigram in contrastive_markers:
                    marker_indices.append(i)
        
        return marker_indices
    
    def lexicon_based_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform lexicon-based sentiment analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and details
        """
        if not text or not self.sentiment_lexicon:
            return {"score": 0.0, "magnitude": 0.0, "label": "neutral", "details": {}}
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Initialize sentiment tracking
        sentiment_score = 0.0
        sentiment_magnitude = 0.0
        sentiment_words = {}
        token_sentiments = [0.0] * len(doc)
        
        # Get negation spans
        negation_spans = []
        contrastive_markers = []
        if self.negation_handling:
            negation_spans = self.detect_negations(doc)
            contrastive_markers = self.detect_contrastive_markers(doc)
        
        # Calculate sentiment based on lexicon
        for i, token in enumerate(doc):
            # Check if the token is in lexicon (check lemma and lowercase)
            token_text = token.text.lower()
            token_lemma = token.lemma_.lower()
            
            sentiment = 0.0
            matched_form = None
            
            if token_text in self.sentiment_lexicon:
                sentiment = self.sentiment_lexicon[token_text]
                matched_form = token_text
            elif token_lemma in self.sentiment_lexicon:
                sentiment = self.sentiment_lexicon[token_lemma]
                matched_form = token_lemma
                
            # Apply negation if token is in a negation span
            if sentiment != 0.0:
                is_negated = False
                for start_idx, end_idx in negation_spans:
                    if start_idx <= i < end_idx:
                        sentiment = -sentiment  # Invert sentiment for negated terms
                        is_negated = True
                        break
                
                token_sentiments[i] = sentiment
                sentiment_words[token.text] = {
                    "score": float(sentiment),
                    "negated": is_negated,
                    "form_matched": matched_form
                }
                
                sentiment_score += sentiment
                sentiment_magnitude += abs(sentiment)
        
        # Adjust for contrastive markers
        # Typically in French, the clause after the contrastive marker carries more weight
        if contrastive_markers and len(token_sentiments) > 0:
            # For each contrastive marker, find the sentiment before and after
            for marker_idx in contrastive_markers:
                before_marker = token_sentiments[:marker_idx]
                after_marker = token_sentiments[marker_idx+1:]
                
                # Calculate sentiment values for clauses
                before_sentiment = sum(before_marker)
                after_sentiment = sum(after_marker)
                
                # If contrasting sentiments, give more weight to the latter
                if (before_sentiment * after_sentiment < 0 or 
                    (before_sentiment == 0 and after_sentiment != 0) or
                    (before_sentiment != 0 and after_sentiment == 0)):
                    
                    # Reduce the influence of sentiment before the marker
                    sentiment_score = sentiment_score - before_sentiment*0.5 + after_sentiment*0.5
        
        # Normalize scores for text length
        if len(doc) > 0:
            normalized_score = sentiment_score / len(doc)
        else:
            normalized_score = 0.0
        
        # Determine sentiment label
        if normalized_score > 0.05:
            label = "positive"
        elif normalized_score < -0.05:
            label = "negative"
        else:
            label = "neutral"
            
        return {
            "score": float(normalized_score),
            "raw_score": float(sentiment_score),
            "magnitude": float(sentiment_magnitude),
            "label": label,
            "details": sentiment_words
        }
    
    def transformer_based_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform transformer-based sentiment analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and details
        """
        if not text:
            return {"score": 0.0, "magnitude": 0.0, "label": "neutral", "details": {}}
        
        if not TRANSFORMERS_AVAILABLE or self.transformer_pipeline is None:
            logger.warning("Transformer-based analysis not available, falling back to lexicon-based")
            return self.lexicon_based_sentiment(text)
        
        try:
            # Run the transformer pipeline on the input text
            results = self.transformer_pipeline(text)[0]
            
            # Extract scores based on the model's output format
            sentiment_scores = {}
            for result in results:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
            
            # The multilingual model uses star ratings (1-5)
            # Convert these to a normalized sentiment score between -1 and 1
            if "1 star" in sentiment_scores or "5 stars" in sentiment_scores:
                # Calculate weighted average score (1=very negative, 3=neutral, 5=very positive)
                weighted_score = 0.0
                total_weight = 0.0
                
                # Map star ratings to sentiment values: 1->-1.0, 2->-0.5, 3->0, 4->0.5, 5->1.0
                star_sentiment_map = {
                    "1 star": -1.0,
                    "2 stars": -0.5,
                    "3 stars": 0.0,
                    "4 stars": 0.5,
                    "5 stars": 1.0
                }
                
                for label, score in sentiment_scores.items():
                    if label in star_sentiment_map:
                        weighted_score += star_sentiment_map[label] * score
                        total_weight += score
                
                if total_weight > 0:
                    normalized_score = weighted_score / total_weight
                else:
                    normalized_score = 0.0
                
                # Find the highest scoring label
                max_label = max(sentiment_scores.items(), key=lambda x: x[1])
                star_label = max_label[0]
                
                # Convert star rating to standard sentiment label
                if star_label in ["4 stars", "5 stars"]:
                    label = "positive"
                elif star_label in ["1 star", "2 stars"]:
                    label = "negative"
                else:
                    label = "neutral"
                
                # Magnitude is the confidence in the prediction
                magnitude = max_label[1]
                
                return {
                    "score": float(normalized_score),
                    "magnitude": float(magnitude),
                    "label": label,
                    "details": sentiment_scores
                }
            else:
                # Handle models with standard positive/negative labels
                # Determine the overall sentiment
                if 'positive' in sentiment_scores and 'negative' in sentiment_scores:
                    # Calculate a normalized score between -1 and 1
                    # 1 = fully positive, -1 = fully negative
                    positive_score = sentiment_scores.get('positive', 0)
                    negative_score = sentiment_scores.get('negative', 0)
                    
                    # Normalized score: difference between positive and negative
                    normalized_score = positive_score - negative_score
                    
                    # Magnitude is the sum of absolute values
                    magnitude = positive_score + negative_score
                    
                    # Determine sentiment label
                    if normalized_score > 0.1:
                        label = "positive"
                    elif normalized_score < -0.1:
                        label = "negative"
                    else:
                        label = "neutral"
                    
                    return {
                        "score": float(normalized_score),
                        "magnitude": float(magnitude),
                        "label": label,
                        "details": sentiment_scores
                    }
                else:
                    # Handle models with different output formats
                    # Just take the highest scoring label
                    max_label = max(sentiment_scores.items(), key=lambda x: x[1])
                    label = max_label[0]
                    score = max_label[1]
                    
                    # Convert to our standard format
                    normalized_score = 0.0
                    if label == "positive":
                        normalized_score = score
                    elif label == "negative":
                        normalized_score = -score
                    
                    return {
                        "score": float(normalized_score),
                        "magnitude": float(score),
                        "label": label,
                        "details": sentiment_scores
                    }
                
        except Exception as e:
            logger.error(f"Error in transformer-based sentiment analysis: {e}")
            # Fall back to lexicon-based as a backup
            return self.lexicon_based_sentiment(text)
    
    def analyze_sentiment(self, text: str, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze sentiment in a text segment.
        
        Args:
            text: Text segment to analyze
            method: Override default method if provided
            
        Returns:
            Dictionary with sentiment analysis results
        """
        method = method or self.method
        
        if method == "lexicon_based":
            return self.lexicon_based_sentiment(text)
        elif method == "transformer_based":
            if TRANSFORMERS_AVAILABLE and self.transformer_pipeline is not None:
                return self.transformer_based_sentiment(text)
            else:
                logger.warning("Transformer-based analysis not available, falling back to lexicon-based")
                return self.lexicon_based_sentiment(text)
        else:
            logger.warning(f"Unsupported method: {method}, falling back to lexicon-based analysis")
            return self.lexicon_based_sentiment(text)
    
    def process_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single text segment for sentiment analysis.
        
        Args:
            segment: Text segment to analyze
            
        Returns:
            Segment with added sentiment metadata
        """
        # Extract text from segment
        text = segment.get("text", "")
        if isinstance(text, list):
            # Join if text is a list of sentences
            text = " ".join(text)
            
        if not text:
            logger.warning(f"Empty text in segment {segment.get('segment_id', 'unknown')}")
            segment["sentiment"] = {"score": 0.0, "magnitude": 0.0, "label": "neutral"}
            return segment
            
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Add sentiment to segment
        segment["sentiment"] = sentiment
        
        return segment
    
    def process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple text segments for sentiment analysis.
        
        Args:
            segments: List of text segments to analyze
            
        Returns:
            Segments with added sentiment metadata
        """
        if not segments:
            return []
            
        # Process each segment
        processed_segments = []
        for segment in segments:
            processed = self.process_segment(segment)
            processed_segments.append(processed)
            
        logger.info(f"Processed {len(processed_segments)} segments for sentiment analysis")
        return processed_segments

