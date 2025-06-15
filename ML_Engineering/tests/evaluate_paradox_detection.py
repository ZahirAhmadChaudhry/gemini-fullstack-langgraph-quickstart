#!/usr/bin/env python
"""
Paradox Detection Evaluation Script

This script evaluates the performance of the paradox detection component against
a set of test cases.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Add parent directory to path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from baseline_nlp.paradox_detection import ParadoxDetector
import baseline_nlp.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParadoxEvaluator:
    """
    Evaluates the performance of the paradox detection component.
    """
    
    def __init__(self, detector=None):
        """
        Initialize the evaluator with a paradox detector instance.
        
        Args:
            detector: An instance of ParadoxDetector. If None, a new instance will be created.
        """
        if detector is None:
            logger.info("Initializing ParadoxDetector...")
            self.detector = ParadoxDetector(
                antonyms_path=config.PARADOX_DETECTION["antonyms_path"],
                tension_keywords_path=config.PARADOX_DETECTION["tension_keywords_path"],
                confidence_threshold=config.PARADOX_DETECTION["confidence_threshold"],
                spacy_model=config.NLP_PIPELINE["spacy_model"]
            )
        else:
            self.detector = detector
        
        self.results = []
    
    def evaluate_test_cases(self, test_cases):
        """
        Evaluate the paradox detector on a list of test cases.
        
        Args:
            test_cases: List of dictionaries, each with 'text', 'expected_is_paradox', and optional 'category', 'id'
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Evaluating {len(test_cases)} test cases...")
        
        results = []
        
        for case in test_cases:
            case_id = case.get('id', f"case_{len(results) + 1}")
            text = case['text']
            expected_is_paradox = case['expected_is_paradox']
            category = case.get('category', 'unknown')
            
            # Run the detector
            paradox_result = self.detector.detect_paradoxes(text)
            
            # Record the results
            result = {
                'id': case_id,
                'text': text,
                'category': category,
                'expected_is_paradox': expected_is_paradox,
                'detected_is_paradox': paradox_result['is_paradox'],
                'confidence': paradox_result['confidence'],
                'num_detections': len(paradox_result['detections']),
                'detection_rules': [d['rule'] for d in paradox_result['detections']],
                'correct': expected_is_paradox == paradox_result['is_paradox']
            }
            
            results.append(result)
        
        # Convert results to DataFrame for easier analysis
        self.results = pd.DataFrame(results)
        return self.results
    
    def calculate_metrics(self):
        """
        Calculate performance metrics based on evaluation results.
        
        Returns:
            Dictionary with metrics (precision, recall, f1, etc.)
        """
        if len(self.results) == 0:
            logger.warning("No results available to calculate metrics")
            return {}
        
        # Basic counts
        true_positives = sum((self.results['expected_is_paradox'] == True) & 
                             (self.results['detected_is_paradox'] == True))
        
        false_positives = sum((self.results['expected_is_paradox'] == False) & 
                              (self.results['detected_is_paradox'] == True))
        
        true_negatives = sum((self.results['expected_is_paradox'] == False) & 
                             (self.results['detected_is_paradox'] == False))
        
        false_negatives = sum((self.results['expected_is_paradox'] == True) & 
                              (self.results['detected_is_paradox'] == False))
        
        # Calculate metrics
        total = len(self.results)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # Metrics by rule type
        rule_counts = {}
        for rules in self.results['detection_rules']:
            for rule in rules:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        # Metrics by category
        category_metrics = {}
        for category in self.results['category'].unique():
            category_subset = self.results[self.results['category'] == category]
            category_total = len(category_subset)
            if category_total > 0:
                category_true_positives = sum((category_subset['expected_is_paradox'] == True) & 
                                              (category_subset['detected_is_paradox'] == True))
                category_false_positives = sum((category_subset['expected_is_paradox'] == False) & 
                                              (category_subset['detected_is_paradox'] == True))
                category_true_negatives = sum((category_subset['expected_is_paradox'] == False) & 
                                             (category_subset['detected_is_paradox'] == False))
                category_false_negatives = sum((category_subset['expected_is_paradox'] == True) & 
                                              (category_subset['detected_is_paradox'] == False))
                
                category_accuracy = (category_true_positives + category_true_negatives) / category_total
                
                if (category_true_positives + category_false_positives) > 0:
                    category_precision = category_true_positives / (category_true_positives + category_false_positives)
                else:
                    category_precision = 0
                    
                if (category_true_positives + category_false_negatives) > 0:
                    category_recall = category_true_positives / (category_true_positives + category_false_negatives)
                else:
                    category_recall = 0
                    
                if (category_precision + category_recall) > 0:
                    category_f1 = 2 * (category_precision * category_recall) / (category_precision + category_recall)
                else:
                    category_f1 = 0
                
                category_metrics[category] = {
                    'accuracy': category_accuracy,
                    'precision': category_precision,
                    'recall': category_recall,
                    'f1': category_f1,
                    'true_positives': int(category_true_positives),
                    'false_positives': int(category_false_positives),
                    'true_negatives': int(category_true_negatives),
                    'false_negatives': int(category_false_negatives),
                    'total': category_total
                }
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': false_positive_rate,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'total': total,
            'rule_counts': rule_counts,
            'by_category': category_metrics
        }
        
        return metrics
    
    def export_results(self, output_dir=None, prefix='paradox_evaluation'):
        """
        Export evaluation results to CSV and metrics to JSON.
        
        Args:
            output_dir: Directory to save output files (defaults to current directory)
            prefix: Prefix for output filenames
            
        Returns:
            Tuple of paths to saved files (results_path, metrics_path)
        """
        if len(self.results) == 0:
            logger.warning("No results available to export")
            return None, None
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export results to CSV
        results_filename = f"{prefix}_results_{timestamp}.csv"
        results_path = os.path.join(output_dir, results_filename)
        self.results.to_csv(results_path, index=False)
        logger.info(f"Exported results to {results_path}")
        
        # Export metrics to JSON
        metrics = self.calculate_metrics()
        metrics_filename = f"{prefix}_metrics_{timestamp}.json"
        metrics_path = os.path.join(output_dir, metrics_filename)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Exported metrics to {metrics_path}")
        
        return results_path, metrics_path

def load_test_cases(filepath):
    """
    Load test cases from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of test case dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    logger.info(f"Loaded {len(test_cases)} test cases from {filepath}")
    return test_cases

def create_sample_test_cases():
    """
    Create a set of sample test cases for demonstration purposes.
    
    Returns:
        List of test case dictionaries
    """
    test_cases = [
        {
            'id': 'antonym_1',
            'text': "L'entreprise vise à augmenter sa production tout en diminuant son empreinte environnementale.",
            'expected_is_paradox': True,
            'category': 'antonym_pair'
        },
        {
            'id': 'antonym_2',
            'text': "Notre objectif est d'accroître les profits à court terme mais de réduire l'impact écologique à long terme.",
            'expected_is_paradox': True,
            'category': 'antonym_pair'
        },
        {
            'id': 'negation_1',
            'text': "Nous croyons au développement durable, mais nous ne croyons pas que cela nécessite de sacrifier la croissance économique.",
            'expected_is_paradox': True,
            'category': 'negated_repetition'
        },
        {
            'id': 'tension_1',
            'text': "La transition vers une économie verte génère des tensions entre le besoin immédiat d'emplois et la nécessité de protéger l'environnement.",
            'expected_is_paradox': True,
            'category': 'sustainability_tension'
        },
        {
            'id': 'contrastive_1',
            'text': "D'une part, nous devons réduire nos émissions de carbone, d'autre part, nous devons maintenir la compétitivité de notre industrie.",
            'expected_is_paradox': True,
            'category': 'contrastive_structure'
        },
        {
            'id': 'non_paradox_1',
            'text': "Les énergies renouvelables offrent des solutions durables pour notre avenir énergétique.",
            'expected_is_paradox': False,
            'category': 'non_paradox'
        },
        {
            'id': 'non_paradox_2',
            'text': "Notre entreprise a réduit ses émissions de carbone de 15% cette année grâce à des investissements dans l'efficacité énergétique.",
            'expected_is_paradox': False,
            'category': 'non_paradox'
        },
        {
            'id': 'borderline_1',
            'text': "Le développement durable exige des compromis entre différents objectifs.",
            'expected_is_paradox': False,  # Subtle mention of trade-offs but not explicit enough
            'category': 'borderline'
        }
    ]
    
    return test_cases

def save_test_cases(test_cases, filepath):
    """
    Save test cases to a JSON file.
    
    Args:
        test_cases: List of test case dictionaries
        filepath: Path to save the JSON file
        
    Returns:
        Path to the saved file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(test_cases)} test cases to {filepath}")
    return filepath

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description='Evaluate paradox detection')
    
    parser.add_argument('--test-cases', '-t',
                        help='Path to test cases JSON file')
    
    parser.add_argument('--create-sample', '-s', action='store_true',
                        help='Create a sample test cases file instead of evaluating')
    
    parser.add_argument('--output-dir', '-o',
                        help='Directory to save output files (defaults to current directory)')
    
    parser.add_argument('--log-level', '-l',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO',
                       help='Set the logging level')
    
    return parser.parse_args()

def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample test cases if requested
    if args.create_sample:
        test_cases = create_sample_test_cases()
        output_dir = args.output_dir or os.path.join(os.getcwd(), 'tests', 'data')
        filepath = os.path.join(output_dir, 'sample_test_cases.json')
        save_test_cases(test_cases, filepath)
        logger.info(f"Created sample test cases file at {filepath}")
        return 0
    
    # Load test cases
    if args.test_cases:
        test_cases = load_test_cases(args.test_cases)
    else:
        # Use sample test cases as default
        logger.info("No test cases file provided, using sample test cases")
        test_cases = create_sample_test_cases()
    
    # Initialize evaluator
    evaluator = ParadoxEvaluator()
    
    # Evaluate test cases
    results = evaluator.evaluate_test_cases(test_cases)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics()
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total test cases: {metrics['total']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    # Export results
    output_dir = args.output_dir or os.path.join(os.getcwd(), 'tests', 'output')
    evaluator.export_results(output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())