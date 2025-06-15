"""
Hybrid Pipeline Integration Package

This package contains components for integrating machine learning models
with the existing rule-based classification system.

Integration Strategy:
- Keep high-performing rule-based components (temporal, codes, concepts)
- Replace underperforming components with ML models (tension, thematic)
- Implement confidence-based decision logic for fallback mechanisms
- Maintain exact compatibility with data.json output format

Components:
- hybrid_classifier.py: Main hybrid classification pipeline
- confidence_manager.py: Confidence threshold management and decision logic
- fallback_systems.py: Rule-based fallback mechanisms for low-confidence predictions
"""

__version__ = "1.0.0"

from typing import Dict, Any, Optional, Tuple

# Confidence thresholds for decision logic
CONFIDENCE_THRESHOLDS = {
    'tension_detection': {
        'high_confidence': 0.8,    # Use ML prediction directly
        'medium_confidence': 0.6,  # Use ML with validation
        'low_confidence': 0.4      # Fall back to rules
    },
    'thematic_classification': {
        'high_confidence': 0.7,    # Use ML prediction directly
        'medium_confidence': 0.5,  # Use ML with validation
        'low_confidence': 0.3      # Fall back to rules
    }
}

# Component configuration
COMPONENT_CONFIG = {
    "rule_based_components": [
        "temporal_classification",    # 100% accuracy - keep unchanged
        "specialized_code_assignment", # 100% accuracy - keep unchanged
        "conceptual_classification"   # 100% accuracy - keep unchanged
    ],
    "ml_enhanced_components": [
        "tension_detection",         # 33% → 75-90% target
        "thematic_classification"    # Limited → 85-95% target
    ]
}

class HybridDecisionLogic:
    """Base class for hybrid ML/rule-based decision logic."""
    
    def __init__(self, confidence_thresholds: Dict[str, Dict[str, float]]):
        self.thresholds = confidence_thresholds
    
    def should_use_ml_prediction(self, confidence: float, component: str) -> bool:
        """Determine whether to use ML prediction based on confidence."""
        threshold = self.thresholds[component]['low_confidence']
        return confidence >= threshold
    
    def get_decision_rationale(self, confidence: float, component: str) -> str:
        """Get human-readable rationale for decision."""
        thresholds = self.thresholds[component]
        
        if confidence >= thresholds['high_confidence']:
            return f"High confidence ({confidence:.3f}) - using ML prediction"
        elif confidence >= thresholds['medium_confidence']:
            return f"Medium confidence ({confidence:.3f}) - using ML with validation"
        elif confidence >= thresholds['low_confidence']:
            return f"Low confidence ({confidence:.3f}) - using ML with caution"
        else:
            return f"Very low confidence ({confidence:.3f}) - falling back to rules"

def validate_integration_compatibility(ml_output: Dict[str, Any], 
                                     rule_output: Dict[str, Any]) -> bool:
    """Validate that ML and rule-based outputs are compatible."""
    required_fields = [
        "Concepts de 2nd ordre",
        "Items de 1er ordre reformulé", 
        "Items de 1er ordre (intitulé d'origine)",
        "Détails",
        "Période",
        "Thème",
        "Code spé"
    ]
    
    # Check that both outputs have required fields
    ml_fields = set(ml_output.keys())
    rule_fields = set(rule_output.keys())
    
    return all(field in ml_fields for field in required_fields) and \
           all(field in rule_fields for field in required_fields)
