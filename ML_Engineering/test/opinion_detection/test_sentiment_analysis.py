"""
Test script for opinion detection module.

This script tests the different sentiment analysis methods implemented in the
SentimentAnalyzer class, including lexicon-based and transformer-based approaches.
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path to import the module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from baseline_nlp.opinion_detection.sentiment_analysis import SentimentAnalyzer

def test_lexicon_based_sentiment(text, lexicon_path, negation_handling=True):
    """
    Test lexicon-based sentiment analysis with the given text.
    
    Args:
        text: Text to analyze
        lexicon_path: Path to sentiment lexicon
        negation_handling: Whether to handle negations
        
    Returns:
        Sentiment analysis results
    """
    analyzer = SentimentAnalyzer(
        method="lexicon_based",
        lexicon_path=lexicon_path,
        negation_handling=negation_handling
    )
    
    return analyzer.analyze_sentiment(text)

def test_transformer_based_sentiment(text, lexicon_path, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Test transformer-based sentiment analysis with the given text.
    
    Args:
        text: Text to analyze
        lexicon_path: Path to sentiment lexicon for fallback
        model_name: Hugging Face model name
        
    Returns:
        Sentiment analysis results
    """
    try:
        analyzer = SentimentAnalyzer(
            method="transformer_based",
            transformer_model=model_name,
            lexicon_path=lexicon_path  # Provide lexicon path for fallback
        )
        
        return analyzer.analyze_sentiment(text)
    except Exception as e:
        logger.error(f"Error in transformer-based analysis: {e}")
        logger.info("Falling back to lexicon-based analysis for testing")
        
        # Fall back to lexicon-based if transformer fails
        return test_lexicon_based_sentiment(text, lexicon_path)

def test_negation_handling(text_with_negation, text_without_negation, lexicon_path):
    """
    Test negation handling by comparing analysis of text with and without negation.
    
    Args:
        text_with_negation: Text containing negation
        text_without_negation: Similar text without negation
        lexicon_path: Path to sentiment lexicon
        
    Returns:
        Tuple of (result with negation handling, result without negation handling)
    """
    # With negation handling
    with_handling = test_lexicon_based_sentiment(
        text_with_negation, lexicon_path, negation_handling=True
    )
    
    # Without negation handling
    without_handling = test_lexicon_based_sentiment(
        text_with_negation, lexicon_path, negation_handling=False
    )
    
    # For comparison, analyze the text without negation
    positive_text = test_lexicon_based_sentiment(
        text_without_negation, lexicon_path, negation_handling=True
    )
    
    return with_handling, without_handling, positive_text

def test_contrastive_markers(text_with_contrast, lexicon_path):
    """
    Test contrastive marker handling.
    
    Args:
        text_with_contrast: Text containing contrastive markers
        lexicon_path: Path to sentiment lexicon
        
    Returns:
        Sentiment analysis results
    """
    return test_lexicon_based_sentiment(text_with_contrast, lexicon_path)

def main():
    """
    Main test function.
    """
    # Path to sample lexicon
    lexicon_path = os.path.join(os.path.dirname(__file__), "sample_lexicon.csv")
    
    # Ensure sample lexicon exists
    if not os.path.exists(lexicon_path):
        # Create a simple sample lexicon for testing
        with open(lexicon_path, 'w', encoding='utf-8') as f:
            f.write("term,polarity\n")
            f.write("bon,positive\n")
            f.write("mauvais,negative\n")
            f.write("excellent,positive\n")
            f.write("terrible,negative\n")
            f.write("efficace,positive\n")
            f.write("inefficace,negative\n")
            f.write("durable,positive\n")
            f.write("polluant,negative\n")
            f.write("écologique,positive\n")
            f.write("nuisible,negative\n")
            f.write("amélioration,positive\n")
            f.write("détérioration,negative\n")
            f.write("avantage,positive\n")
            f.write("inconvénient,negative\n")
            f.write("réussite,positive\n")
            f.write("échec,negative\n")
            f.write("progrès,positive\n")
            f.write("recul,negative\n")
            f.write("solution,positive\n")
            f.write("problème,negative\n")
            f.write("opportunité,positive\n")
            f.write("menace,negative\n")
            f.write("bénéfique,positive\n")
            f.write("nocif,negative\n")
            f.write("favorable,positive\n")
            f.write("défavorable,negative\n")
            f.write("protection,positive\n")
            f.write("destruction,negative\n")
            f.write("préserver,positive\n")
            f.write("détruire,negative\n")
        logger.info(f"Created sample lexicon at {lexicon_path}")
    
    # Sample French texts related to sustainability
    test_texts = [
        {
            "title": "Positive Sustainability Text",
            "text": "Les énergies renouvelables sont une solution efficace et durable pour lutter contre le changement climatique. "
                   "Elles offrent de nombreux avantages pour l'environnement et créent des opportunités économiques."
        },
        {
            "title": "Negative Sustainability Text",
            "text": "La pollution industrielle est un problème majeur qui menace nos écosystèmes. "
                   "Les émissions de carbone causent des dommages terribles et la situation continue à se détériorer."
        },
        {
            "title": "Mixed Sentiment Text",
            "text": "Le développement des énergies propres progresse, mais les défis restent nombreux. "
                   "Malgré quelques réussites notables, beaucoup d'obstacles persistent."
        }
    ]
    
    # Test cases for negation handling
    negation_test_pairs = [
        {
            "with_negation": "Cette solution n'est pas efficace pour réduire la pollution.",
            "without_negation": "Cette solution est efficace pour réduire la pollution."
        },
        {
            "with_negation": "Ces mesures ne sont jamais bénéfiques pour l'environnement.",
            "without_negation": "Ces mesures sont bénéfiques pour l'environnement."
        }
    ]
    
    # Test cases for contrastive markers
    contrast_test_texts = [
        "Le projet est écologique mais il coûte cher.",
        "Les énergies renouvelables sont prometteuses, cependant leur intermittence reste un défi.",
        "Bien que coûteux, l'investissement est nécessaire pour protéger l'environnement."
    ]
    
    print("Testing Opinion Detection Module\n")
    
    # Test 1: Lexicon-based sentiment analysis
    print("\n=== Test 1: Lexicon-Based Sentiment Analysis ===\n")
    for test_case in test_texts:
        print(f"Text: {test_case['title']}")
        print("-" * 40)
        print(f"{test_case['text'][:100]}...")
        print("-" * 40)
        
        # Analyze sentiment
        sentiment = test_lexicon_based_sentiment(test_case["text"], lexicon_path)
        
        # Print results
        print(f"Sentiment Label: {sentiment['label']}")
        print(f"Sentiment Score: {sentiment['score']:.4f}")
        print(f"Sentiment Magnitude: {sentiment['magnitude']:.4f}")
        
        # Print detected sentiment words
        print("\nDetected Sentiment Words:")
        for word, details in sentiment["details"].items():
            print(f"  {word}: Score={details['score']:.2f}, Negated={details['negated']}")
        
        print("\n" + "-" * 40 + "\n")
    
    # Test 2: Negation Handling
    print("\n=== Test 2: Negation Handling ===\n")
    for i, test_pair in enumerate(negation_test_pairs):
        print(f"Test Pair {i+1}:")
        print(f"Text with negation: {test_pair['with_negation']}")
        print(f"Text without negation: {test_pair['without_negation']}")
        print("-" * 40)
        
        # Test negation handling
        with_handling, without_handling, positive_text = test_negation_handling(
            test_pair["with_negation"], 
            test_pair["without_negation"],
            lexicon_path
        )
        
        # Print results
        print("Results with negation handling:")
        print(f"  Label: {with_handling['label']}")
        print(f"  Score: {with_handling['score']:.4f}")
        
        print("\nResults without negation handling:")
        print(f"  Label: {without_handling['label']}")
        print(f"  Score: {without_handling['score']:.4f}")
        
        print("\nResults for positive text (for comparison):")
        print(f"  Label: {positive_text['label']}")
        print(f"  Score: {positive_text['score']:.4f}")
        
        print("\n" + "-" * 40 + "\n")
    
    # Test 3: Contrastive Marker Handling
    print("\n=== Test 3: Contrastive Marker Handling ===\n")
    for i, text in enumerate(contrast_test_texts):
        print(f"Text {i+1}: {text}")
        print("-" * 40)
        
        # Test contrastive marker handling
        sentiment = test_contrastive_markers(text, lexicon_path)
        
        # Print results
        print(f"Sentiment Label: {sentiment['label']}")
        print(f"Sentiment Score: {sentiment['score']:.4f}")
        
        print("\n" + "-" * 40 + "\n")
    
    # Test 4: Transformer-based sentiment analysis (if available)
    print("\n=== Test 4: Transformer-Based Sentiment Analysis ===\n")
    try:
        for test_case in test_texts:
            print(f"Text: {test_case['title']}")
            print("-" * 40)
            print(f"{test_case['text'][:100]}...")
            print("-" * 40)
            
            # Analyze sentiment with the transformer method (with lexicon for fallback)
            sentiment = test_transformer_based_sentiment(test_case["text"], lexicon_path)
            
            # Print results
            print(f"Sentiment Label: {sentiment['label']}")
            print(f"Sentiment Score: {sentiment['score']:.4f}")
            print(f"Sentiment Magnitude: {sentiment['magnitude']:.4f}")
            
            # Print sentiment details
            print("\nSentiment Details:")
            if isinstance(sentiment["details"], dict):
                if all(isinstance(v, float) for v in sentiment["details"].values()):
                    # For transformer results
                    for label, score in sentiment["details"].items():
                        print(f"  {label}: {score:.4f}")
                else:
                    # For lexicon-based fallback results
                    for word, details in sentiment["details"].items():
                        if isinstance(details, dict) and "score" in details:
                            print(f"  {word}: Score={details['score']:.2f}, Negated={details.get('negated', False)}")
            
            print("\n" + "-" * 40 + "\n")
    except Exception as e:
        print(f"Error testing transformer-based sentiment analysis: {e}")
        print("Skipping transformer-based tests. Please install transformers library if needed.")
    
    print("All tests completed!")

if __name__ == "__main__":
    main()