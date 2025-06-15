#!/usr/bin/env python3
"""
Classification Models Performance Analysis

This script analyzes the classification models used in the ML pipeline
and evaluates their performance against the target format requirements.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_classification_models():
    """Analyze the classification models used in the ML pipeline."""
    
    logger.info("🔍 Analyzing Classification Models in ML Pipeline...")
    
    # Load test results
    target_format_path = "test_output/test_target_format_integration_target_format.json"
    processed_segments_path = "test_output/test_target_format_integration_processed_segments.json"
    
    if not os.path.exists(target_format_path):
        logger.error("Target format test results not found. Run test_target_format_integration.py first.")
        return
    
    with open(target_format_path, 'r', encoding='utf-8') as f:
        target_format_data = json.load(f)
    
    with open(processed_segments_path, 'r', encoding='utf-8') as f:
        processed_segments = json.load(f)
    
    # Load reference data for comparison
    reference_path = "data_from_Data_Engineering/_qHln3fOjOg_target_format.json"
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    logger.info(f"Loaded {len(target_format_data['entries'])} generated entries")
    logger.info(f"Loaded {len(reference_data['entries'])} reference entries")
    logger.info(f"Loaded {len(processed_segments)} processed segments")
    
    # Analyze each classification model
    results = {
        "temporal_classification": analyze_temporal_classification(target_format_data, reference_data, processed_segments),
        "thematic_classification": analyze_thematic_classification(target_format_data, reference_data, processed_segments),
        "conceptual_classification": analyze_conceptual_classification(target_format_data, reference_data, processed_segments),
        "tension_detection": analyze_tension_detection(target_format_data, reference_data, processed_segments),
        "specialized_code_assignment": analyze_specialized_codes(target_format_data, reference_data, processed_segments)
    }
    
    # Generate comprehensive report
    generate_performance_report(results)
    
    return results


def analyze_temporal_classification(generated_data, reference_data, processed_segments):
    """Analyze temporal classification performance."""
    logger.info("📅 Analyzing Temporal Classification...")
    
    # Extract temporal predictions and ground truth
    generated_periods = [entry["Période"] for entry in generated_data["entries"]]
    reference_periods = [entry["Période"] for entry in reference_data["entries"][:len(generated_periods)]]
    
    # Calculate accuracy
    correct_predictions = sum(1 for g, r in zip(generated_periods, reference_periods) if g == r)
    accuracy = correct_predictions / len(generated_periods) if generated_periods else 0
    
    # Analyze temporal features used
    temporal_features = []
    for segment in processed_segments[:len(generated_periods)]:
        ml_features = segment.get("features", {}).get("ml_features", {})
        temporal_features.append({
            "temporal_period": ml_features.get("temporal_period"),
            "temporal_context": segment.get("features", {}).get("temporal_context"),
            "temporal_confidence": segment.get("features", {}).get("temporal_confidence", {})
        })
    
    # Distribution analysis
    period_distribution = {}
    for period in generated_periods:
        period_distribution[period] = period_distribution.get(period, 0) + 1
    
    return {
        "model_type": "Rule-based with ML features",
        "algorithm": "Direct mapping from temporal_period + fallback rules",
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": len(generated_periods),
        "period_distribution": period_distribution,
        "temporal_features_used": temporal_features,
        "performance_target": "90%+ accuracy",
        "achieved_target": accuracy >= 0.9
    }


def analyze_thematic_classification(generated_data, reference_data, processed_segments):
    """Analyze thematic classification performance."""
    logger.info("🎯 Analyzing Thematic Classification...")
    
    # Extract thematic predictions and ground truth
    generated_themes = [entry["Thème"] for entry in generated_data["entries"]]
    reference_themes = [entry["Thème"] for entry in reference_data["entries"][:len(generated_themes)]]
    
    # Calculate accuracy
    correct_predictions = sum(1 for g, r in zip(generated_themes, reference_themes) if g == r)
    accuracy = correct_predictions / len(generated_themes) if generated_themes else 0
    
    # Analyze thematic features used
    thematic_features = []
    for segment in processed_segments[:len(generated_themes)]:
        ml_features = segment.get("features", {}).get("ml_features", {})
        thematic_features.append({
            "performance_score": ml_features.get("performance_score", 0.0),
            "legitimacy_score": ml_features.get("legitimacy_score", 0.0),
            "thematic_indicators": segment.get("features", {}).get("thematic_indicators", {})
        })
    
    # Distribution analysis
    theme_distribution = {}
    for theme in generated_themes:
        theme_distribution[theme] = theme_distribution.get(theme, 0) + 1
    
    return {
        "model_type": "Score-based comparison",
        "algorithm": "performance_score vs legitimacy_score comparison",
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": len(generated_themes),
        "theme_distribution": theme_distribution,
        "thematic_features_used": thematic_features,
        "performance_target": "85-95% accuracy",
        "achieved_target": 0.85 <= accuracy <= 0.95
    }


def analyze_conceptual_classification(generated_data, reference_data, processed_segments):
    """Analyze conceptual classification performance."""
    logger.info("🧠 Analyzing Conceptual Classification...")
    
    # Extract conceptual predictions and ground truth
    generated_concepts = [entry["Concepts de 2nd ordre"] for entry in generated_data["entries"]]
    reference_concepts = [entry["Concepts de 2nd ordre"] for entry in reference_data["entries"][:len(generated_concepts)]]
    
    # Calculate accuracy
    correct_predictions = sum(1 for g, r in zip(generated_concepts, reference_concepts) if g == r)
    accuracy = correct_predictions / len(generated_concepts) if generated_concepts else 0
    
    # Analyze conceptual features used
    conceptual_features = []
    for segment in processed_segments[:len(generated_concepts)]:
        features = segment.get("features", {})
        conceptual_features.append({
            "conceptual_markers": features.get("conceptual_markers", []),
            "conceptual_complexity": features.get("ml_features", {}).get("conceptual_complexity", 0.0)
        })
    
    # Distribution analysis
    concept_distribution = {}
    for concept in generated_concepts:
        concept_distribution[concept] = concept_distribution.get(concept, 0) + 1
    
    return {
        "model_type": "Rule-based with fallback logic",
        "algorithm": "conceptual_markers mapping + tension-based fallback",
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": len(generated_concepts),
        "concept_distribution": concept_distribution,
        "conceptual_features_used": conceptual_features,
        "performance_target": "High-quality generation",
        "achieved_target": accuracy >= 0.7  # Reasonable threshold for conceptual mapping
    }


def analyze_tension_detection(generated_data, reference_data, processed_segments):
    """Analyze tension detection and mapping performance."""
    logger.info("⚡ Analyzing Tension Detection...")
    
    # Extract tension-based items
    generated_items = [entry["Items de 1er ordre reformulé"] for entry in generated_data["entries"]]
    reference_items = [entry["Items de 1er ordre reformulé"] for entry in reference_data["entries"][:len(generated_items)]]
    
    # Calculate accuracy
    correct_predictions = sum(1 for g, r in zip(generated_items, reference_items) if g == r)
    accuracy = correct_predictions / len(generated_items) if generated_items else 0
    
    # Analyze tension features used
    tension_features = []
    for segment in processed_segments[:len(generated_items)]:
        features = segment.get("features", {})
        tension_features.append({
            "tension_patterns": features.get("tension_patterns", {}),
            "tension_indicators": features.get("ml_features", {}).get("tension_indicators", [])
        })
    
    # Distribution analysis
    items_distribution = {}
    for item in generated_items:
        items_distribution[item] = items_distribution.get(item, 0) + 1
    
    return {
        "model_type": "Pattern-based detection with strength scoring",
        "algorithm": "tension_patterns analysis + strength-based selection",
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": len(generated_items),
        "items_distribution": items_distribution,
        "tension_features_used": tension_features,
        "performance_target": "75-90% accuracy",
        "achieved_target": 0.75 <= accuracy <= 0.90
    }


def analyze_specialized_codes(generated_data, reference_data, processed_segments):
    """Analyze specialized code assignment performance."""
    logger.info("🔧 Analyzing Specialized Code Assignment...")
    
    # Extract specialized codes
    generated_codes = [entry["Code spé"] for entry in generated_data["entries"]]
    reference_codes = [entry["Code spé"] for entry in reference_data["entries"][:len(generated_codes)]]
    
    # Calculate accuracy
    correct_predictions = sum(1 for g, r in zip(generated_codes, reference_codes) if g == r)
    accuracy = correct_predictions / len(generated_codes) if generated_codes else 0
    
    # Distribution analysis
    code_distribution = {}
    for code in generated_codes:
        code_distribution[code] = code_distribution.get(code, 0) + 1
    
    return {
        "model_type": "Mapping-based assignment",
        "algorithm": "tension-to-code mapping + theme-based fallback",
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": len(generated_codes),
        "code_distribution": code_distribution,
        "performance_target": "Robust assignment",
        "achieved_target": accuracy >= 0.8  # High threshold for code consistency
    }


def generate_performance_report(results):
    """Generate a comprehensive performance report."""
    logger.info("📊 Generating Performance Report...")
    
    print("\n" + "="*80)
    print("🎯 CLASSIFICATION MODELS PERFORMANCE ANALYSIS")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n📈 {model_name.upper().replace('_', ' ')}")
        print("-" * 60)
        print(f"Model Type: {model_results['model_type']}")
        print(f"Algorithm: {model_results['algorithm']}")
        print(f"Accuracy: {model_results['accuracy']:.2%}")
        print(f"Correct Predictions: {model_results['correct_predictions']}/{model_results['total_predictions']}")
        print(f"Performance Target: {model_results['performance_target']}")
        print(f"Target Achieved: {'✅ YES' if model_results['achieved_target'] else '❌ NO'}")
        
        if 'distribution' in str(model_results):
            for key, value in model_results.items():
                if 'distribution' in key:
                    print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Overall summary
    total_models = len(results)
    achieved_targets = sum(1 for r in results.values() if r['achieved_target'])
    overall_success_rate = achieved_targets / total_models
    
    print(f"\n🏆 OVERALL PERFORMANCE SUMMARY")
    print("-" * 60)
    print(f"Models Analyzed: {total_models}")
    print(f"Targets Achieved: {achieved_targets}/{total_models}")
    print(f"Overall Success Rate: {overall_success_rate:.2%}")
    
    if overall_success_rate >= 0.8:
        print("✅ EXCELLENT: ML Pipeline classification performance is strong!")
    elif overall_success_rate >= 0.6:
        print("⚠️  GOOD: ML Pipeline performance is acceptable with room for improvement")
    else:
        print("❌ NEEDS IMPROVEMENT: Classification models require enhancement")


def main():
    """Main entry point."""
    logger.info("Starting Classification Models Performance Analysis...")
    
    try:
        results = analyze_classification_models()
        logger.info("✅ Analysis completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
