# Temporal Context Distinction Component

## Overview

The temporal context distinction component classifies French sustainability discourse text segments as referring to present (2023) or future (2050) contexts. It uses rule-based analysis of verb tenses, temporal markers, and other linguistic cues to distinguish the temporal framing of sustainability discussions.

## Algorithm

The implementation employs multiple rule-based detection methods:

1. **Verb Tense Analysis**:
   - Identify and count French verb tenses using spaCy's part-of-speech and morphological analysis
   - Classify tenses as present or future indicators
   - Handle special cases like Présent used for future reference

2. **Temporal Marker Detection**:
   - Identify explicit temporal expressions like "aujourd'hui", "en 2050"
   - Detect relative time expressions ("dans X ans", "à l'avenir")
   - Flag direct references to 2023 or 2050

3. **Discourse Marker Analysis**:
   - Identify discourse markers that signal temporal shifts
   - Detect hypothetical constructions indicating future scenarios
   - Analyze conditional structures

4. **Context Determination**:
   - Weigh evidence from verb tenses and temporal markers
   - Calculate confidence score for temporal classification
   - Assign final temporal context category

## Dependencies

- **Primary Library**: spaCy with fr_core_news_lg model
- **Secondary Libraries**: regex for pattern matching

## Resources

- **Present Markers**: List of French present temporal expressions at `baseline_nlp/temporal_context/resources/present_markers_fr.txt`
- **Future Markers**: List of French future temporal expressions at `baseline_nlp/temporal_context/resources/future_markers_fr.txt`
- **Tense Patterns**: JSON patterns for French verb tense detection at `baseline_nlp/temporal_context/resources/tense_patterns_fr.json`

## Input/Output

### Input
- Text segment (string or list of strings)
- Optional parameters:
  - confidence_threshold: Minimum confidence for classification (default: 0.6)
  - default_context: Default classification when confidence is below threshold (default: "present")

### Output
- Temporal context dictionary with:
  - `context`: String classification as "present" (2023) or "future" (2050)
  - `confidence`: Confidence score between 0 and 1
  - `evidence`: Dictionary containing supporting evidence for the classification

## Example Output

```json
"temporal_context": {
  "context": "future",
  "confidence": 0.85,
  "evidence": {
    "markers": {
      "present": [],
      "future": ["en 2050", "à l'avenir"]
    },
    "tenses": {
      "present": 2,
      "futur_simple": 3,
      "conditionnel": 1,
      "present_for_future": 2
    }
  }
}
```

## Performance Considerations

- **Computational Complexity**: O(n) where n is the number of tokens
- **Memory Usage**: Light, primarily for marker dictionaries
- **Runtime**: Fast, typically milliseconds per segment
- **Resource Size**: ~100 temporal markers, ~50 tense patterns

## Limitations

- Challenge with ambiguous use of Présent tense in French
- Difficulty with implicit temporal framing without explicit markers
- Limited handling of complex temporal references
- No semantic understanding of context beyond rules
- Potential confusion with hypothetical scenarios vs. actual future prediction

## Future Improvements

- Implement more sophisticated tense analysis
- Develop better detection of Présent used for future reference
- Add machine learning components for contextual understanding
- Create domain adaptation for sustainability-specific temporal expressions
- Incorporate broader document context for improved classification
- Add certainty assessment for future predictions vs. hypotheticals