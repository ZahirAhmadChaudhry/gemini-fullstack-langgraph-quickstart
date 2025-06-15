# Paradox Detection Component

## Overview

The paradox detection component identifies linguistic paradoxes in French sustainability discourse. It employs a rule-based approach to detect contradictory statements, tensions, and ambiguities that are common in sustainability discussions.

## Algorithm

The implementation uses multiple rule-based detection methods that operate in parallel:

1. **Antonym Detection**:
   - Identify antonym pairs within proximity thresholds
   - Analyze co-occurrence patterns of opposing terms
   - Map spaCy tokens to the French antonym dictionary

2. **Tension Keywords**:
   - Search for linguistic markers that signal paradoxical thinking
   - Examples: "contradictoire", "paradoxe", "tension", "dilemme"
   - Weight by proximity to sustainability terms

3. **Syntactic Pattern Analysis**:
   - Identify syntactic structures associated with paradoxes
   - Detect contrast structures ("X mais Y", "d'une part... d'autre part")
   - Recognize negation-affirmation patterns

4. **Confidence Scoring**:
   - Combine evidence from multiple detection methods
   - Calculate a confidence score based on rule overlap
   - Apply thresholds for positive detection

## Dependencies

- **Primary Library**: spaCy with fr_core_news_lg model
- **Secondary Libraries**: nltk for text processing, custom pattern matching

## Resources

- **French Antonym Dictionary**: Mappings of French terms to their antonyms at `baseline_nlp/paradox_detection/resources/french_antonyms.json`
- **Tension Keywords**: List of French paradox and tension indicators at `baseline_nlp/paradox_detection/resources/tension_keywords_fr.txt`
- **Syntactic Patterns**: JSON patterns for paradoxical structures at `baseline_nlp/paradox_detection/resources/paradox_patterns_fr.json`
- **Sustainability Terms**: Shared list of sustainability vocabulary at `baseline_nlp/shared_resources/sustainability_terms_fr.txt`

## Input/Output

### Input
- Text segment (string or list of strings)
- Optional parameters:
  - proximity_threshold: Maximum token distance for antonym co-occurrence (default: 15)
  - confidence_threshold: Minimum confidence for paradox detection (default: 0.5)

### Output
- Paradox detection dictionary with:
  - `is_paradox`: Boolean indicating whether a paradox was detected
  - `confidence`: Confidence score between 0 and 1
  - `detections`: List of specific paradox detections with detection method and evidence

## Example Output

```json
"paradox": {
  "is_paradox": true,
  "confidence": 0.75,
  "detections": [
    {
      "method": "antonym_pair",
      "evidence": {
        "term1": "augmentation",
        "term2": "diminution",
        "distance": 8
      }
    },
    {
      "method": "tension_keyword",
      "evidence": {
        "term": "contradictoire",
        "position": 14
      }
    }
  ]
}
```

## Performance Considerations

- **Computational Complexity**: O(n²) in worst case for antonym pair detection
- **Memory Usage**: Light, primarily for resource dictionaries
- **Runtime**: Typically fast for individual segments
- **Dictionary Size**: ~2,000 antonym pairs, ~50 tension keywords

## Limitations

- Rule-based approach lacks semantic understanding
- Limited to explicit paradoxes; misses implicit contradictions
- Fixed dictionaries may not cover domain-specific paradoxes
- May produce false positives with rhetorical devices like contrast
- No learning capability to adapt to new patterns

## Future Improvements

- Expand antonym dictionary with sustainability-specific oppositions
- Implement machine learning for pattern recognition
- Develop semantic understanding of sustainability paradoxes
- Create domain adaptation for specific sustainability sub-topics
- Add context-sensitive confidence adjustments
- Incorporate discourse-level analysis beyond single segments