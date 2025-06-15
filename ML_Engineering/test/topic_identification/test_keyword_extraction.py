"""
Test script for the keyword extraction module.

This script tests the different keyword extraction methods implemented in the
KeywordExtractor class, including TextRank, TF-IDF and YAKE!.
"""

import sys
import os
import json
from pathlib import Path

# Add the parent directory to the Python path to import the module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from baseline_nlp.topic_identification.keyword_extraction import KeywordExtractor

def test_keyword_extraction(text, method="textrank", num_keywords=5):
    """
    Test the keyword extraction with a given text and method.
    
    Args:
        text: Text to extract keywords from
        method: Keyword extraction method to use
        num_keywords: Number of keywords to extract
        
    Returns:
        List of extracted keywords with scores
    """
    # Initialize the keyword extractor with the specified method
    extractor = KeywordExtractor(method=method, num_keywords=num_keywords)
    
    # Extract keywords
    keywords = extractor.extract_keywords(text)
    
    return keywords

def main():
    """
    Main test function that runs the keyword extraction tests.
    """
    print("Testing Keyword Extraction Methods\n")
    
    # Sample French texts related to sustainability
    test_texts = [
        {
            "title": "Climate Change",
            "text": "Le changement climatique représente l'un des plus grands défis de notre époque. "
                   "L'Accord de Paris engage les pays à limiter le réchauffement climatique à bien moins de 2 degrés. "
                   "Les émissions de gaz à effet de serre, notamment le dioxyde de carbone, contribuent à l'effet de serre. "
                   "La transition énergétique vers les énergies renouvelables comme l'éolien et le solaire est essentielle."
        },
        {
            "title": "Circular Economy",
            "text": "L'économie circulaire vise à réduire les déchets et optimiser l'utilisation des ressources. "
                   "Le recyclage, le compostage et la réutilisation sont des pratiques importantes. "
                   "Les entreprises adoptent des stratégies d'écoconception pour minimiser l'impact environnemental "
                   "de leurs produits. La gestion durable des déchets est cruciale pour la protection de l'environnement."
        },
        {
            "title": "Biodiversity",
            "text": "La biodiversité est menacée par les activités humaines comme la déforestation et la pollution. "
                   "La protection des écosystèmes est essentielle pour maintenir l'équilibre écologique. "
                   "De nombreuses espèces sont en danger d'extinction, ce qui pourrait avoir des conséquences graves "
                   "pour les écosystèmes et pour les humains. La conservation de la nature est une priorité mondiale."
        }
    ]
    
    # Test each method with each text
    methods = ["textrank", "yake"]
    
    for method in methods:
        print(f"\n=== Testing {method.upper()} method ===\n")
        
        for test_case in test_texts:
            print(f"Text: {test_case['title']}")
            print("-" * 40)
            print(f"{test_case['text'][:100]}...")
            print("-" * 40)
            
            # Extract keywords
            keywords = test_keyword_extraction(test_case["text"], method=method)
            
            # Print results
            print("Extracted Keywords:")
            for i, keyword in enumerate(keywords):
                print(f"{i+1}. {keyword['term']} (score: {keyword['score']:.4f})")
            
            print("\n")
    
    print("All tests completed!")

if __name__ == "__main__":
    main()