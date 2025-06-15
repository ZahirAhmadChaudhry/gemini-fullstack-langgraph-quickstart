"""
Configuration settings for the Baseline NLP System for French Sustainability Opinion Analysis.
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT.parent, "preprocessed_data_by_Data_Engineer")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "labeled_output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Topic Identification settings
TOPIC_IDENTIFICATION = {
    "method": "textrank",  # Options: tfidf, rake, yake, textrank, keybert
    "num_keywords": 5,
    "sustainability_terms_path": os.path.join(PROJECT_ROOT, "data", "sustainability_terms_fr.txt"),
}

# Opinion Detection settings
OPINION_DETECTION = {
    "method": "lexicon_based",  # Options: lexicon_based, ml_based, transformer_based
    "lexicon": "FEEL",  # Options: FEEL, LSDfr, UniSent
    "french_sentiment_lexicon_path": os.path.join(PROJECT_ROOT, "data", "french_sentiment_lexicon.csv"),
    "negation_handling": True,
}

# Paradox Detection settings
PARADOX_DETECTION = {
    "method": "rule_based",
    "sustainability_paradox_terms_path": os.path.join(PROJECT_ROOT, "data", "sustainability_terms_fr.txt"),
    "antonyms_path": os.path.join(PROJECT_ROOT, "data", "french_antonyms.csv"),
    "tension_keywords_path": os.path.join(PROJECT_ROOT, "data", "tension_keywords_fr.csv"),
    "confidence_threshold": 0.6,  # Default threshold for paradox detection
}

# Temporal Context Distinction settings
TEMPORAL_CONTEXT = {
    "method": "rule_based",
    "present_markers_path": os.path.join(PROJECT_ROOT, "data", "present_markers_fr.csv"),
    "future_markers_path": os.path.join(PROJECT_ROOT, "data", "future_markers_fr.csv"),
    "verb_tense_patterns_path": os.path.join(PROJECT_ROOT, "data", "french_verb_tense_patterns.csv"),
}

# NLP Pipeline settings
NLP_PIPELINE = {
    "spacy_model": "fr_core_news_lg",
    "batch_size": 100,
}

# Logging settings
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

