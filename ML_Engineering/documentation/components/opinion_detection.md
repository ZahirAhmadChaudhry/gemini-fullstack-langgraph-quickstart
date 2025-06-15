# Opinion Detection Component

## Overview

The opinion detection component performs sentiment analysis on French sustainability discourse text segments. It employs a lexicon-based approach with French adaptations and linguistic rule handling to identify the sentiment polarity of each text segment.

## Algorithm

The implementation utilizes a lexicon-based approach enhanced with rules for French linguistic patterns:

1. **Preprocessing**:
   - Text normalization (lowercasing, accent normalization)
   - Tokenization using spaCy's French language model
   - Lemmatization of tokens to match with lexicon

2. **Lexicon Matching**:
   - Look up terms in the French sentiment lexicon
   - Assign polarity scores to matched terms
   - Calculate raw sentiment score based on lexicon matches

3. **Linguistic Rule Application**:
   - Detect and handle negation (e.g., "ne...pas", "ne...plus", "jamais")
   - Process intensifiers and diminishers (e.g., "très", "peu")
   - Handle contrastive markers (e.g., "mais", "cependant")
   - Account for idiomatic expressions

4. **Sentiment Calculation**:
   - Compute adjusted sentiment score
   - Calculate sentiment magnitude based on intensity/strength
   - Assign sentiment label (positive, negative, neutral)

## Dependencies

- **Primary Library**: spaCy with fr_core_news_lg model
- **Secondary Libraries**: nltk for text preprocessing

## Resources

- **French Sentiment Lexicon**: A lexicon with French words and sentiment polarity scores at `baseline_nlp/opinion_detection/resources/french_sentiment_lexicon.csv`
- **Negation Terms**: List of French negation expressions at `baseline_nlp/opinion_detection/resources/french_negation_terms.txt`
- **Intensifiers/Diminishers**: List of French intensifiers and diminishers at `baseline_nlp/opinion_detection/resources/french_intensifiers.txt`
- **Contrastive Markers**: List of French contrastive markers at `baseline_nlp/opinion_detection/resources/french_contrastive_markers.txt`

## Input/Output

### Input
- Text segment (string or list of strings)
- Optional parameters:
  - use_linguistic_rules: Boolean to enable/disable rule-based adjustments (default: True)
  - threshold: Neutral sentiment threshold (default: 0.1)

### Output
- Sentiment dictionary with:
  - `score`: Normalized sentiment score between -1 and 1 (float)
  - `raw_score`: Unadjusted sentiment score from lexicon matching
  - `magnitude`: Sentiment intensity/strength (float)
  - `label`: Sentiment classification as "positive", "negative", or "neutral"
  - `details`: Dictionary with supporting information about the analysis

## Example Output

```json
"sentiment": {
  "score": -0.25,
  "raw_score": -0.15,
  "magnitude": 0.4,
  "label": "negative",
  "details": {
    "negations": 1,
    "intensifiers": 0,
    "contrastive_markers": 1
  }
}
```

## Performance Considerations

- **Computational Complexity**: O(n) where n is the number of tokens
- **Memory Usage**: Light, primarily for lexicon storage
- **Runtime**: Fast, typically milliseconds per segment
- **Lexicon Size**: ~5,000 entries

## Limitations

- Limited by the coverage of the sentiment lexicon
- Difficulty with domain-specific terminology not in the lexicon
- Challenge handling irony, sarcasm, and implicit sentiment
- Simplified handling of complex linguistic phenomena in French
- No contextual understanding beyond simple rules

## Future Improvements

- Expand the sentiment lexicon with domain-specific sustainability terms
- Implement more sophisticated negation scope detection
- Add machine learning components for context awareness
- Incorporate semantic analysis for implicit sentiment
- Develop better handling of comparative and superlative expressions
- Create adaptation mechanisms for different sustainability topics